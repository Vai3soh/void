/*--------------------------------------------------------------------------------------
 *  Copyright 2025 Glass Devtools, Inc. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt for more information.
 *--------------------------------------------------------------------------------------*/

import { CancellationToken } from '../../../../base/common/cancellation.js';
import { timeout } from '../../../../base/common/async.js';
import { generateUuid } from '../../../../base/common/uuid.js';
import { IFileService } from '../../../../platform/files/common/files.js';
import { IVoidSettingsService } from '../../../../platform/void/common/voidSettingsService.js';
import { ILLMMessageService } from '../common/sendLLMMessageService.js';
import { IToolsService, isDangerousTerminalCommand } from '../common/toolsService.js';
import { ILanguageModelToolsService } from '../../chat/common/languageModelToolsService.js';
import { IMetricsService } from '../../../../platform/void/common/metricsService.js';
import { IConvertToLLMMessageService } from './convertToLLMMessageService.js';
import { LLMLoopDetector, LOOP_DETECTED_MESSAGE } from '../../../../platform/void/common/loopGuard.js';
import { getErrorMessage, RawToolCallObj, RawToolParamsObj, LLMTokenUsage, type OnText } from '../../../../platform/void/common/sendLLMMessageTypes.js';
import { isAToolName } from '../common/prompt/prompts.js';
import { approvalTypeOfToolName, } from '../../../../platform/void/common/toolsServiceTypes.js';
import { ChatMessage, ToolMessage, ChatAttachment } from '../../../../platform/void/common/chatThreadServiceTypes.js';
import { ModelSelection, ModelSelectionOptions } from '../../../../platform/void/common/voidSettingsTypes.js';
import { getModelCapabilities } from '../../../../platform/void/common/modelInference.js';
import { type JsonObject, type JsonValue, isJsonObject, stringifyUnknown, toJsonObject } from '../../../../platform/void/common/jsonTypes.js';
import { classifyToolCall, duplicateWriteTargetError } from '../../../../platform/void/common/toolExecutionPolicy.js';

import { ChatHistoryCompressor, ThreadHistoryCompressionInfo } from './ChatHistoryCompressor.js';
import { ChatToolOutputManager } from './ChatToolOutputManager.js';
import { IThreadStateAccess } from './ChatAcpHandler.js';
import { IMCPService } from '../common/mcpService.js';
import { formatAgentSkillActivationContent } from '../common/skills/agentSkillsPrompt.js';

export type IsRunningType =
	| 'LLM' // the LLM is currently streaming
	| 'tool' // whether a tool is currently running
	| 'awaiting_user' // awaiting user call
	| 'idle' // nothing is running now, but the chat should still appear like it's going (used in-between calls)
	| undefined

const STREAM_UPDATE_INTERVAL_MS = 50;
const LIVE_REASONING_PREVIEW_CHARS = 8_000;

const toLiveReasoningPreview = (reasoning: string): string => {
	if (reasoning.length <= LIVE_REASONING_PREVIEW_CHARS) return reasoning;
	const sectionLength = Math.floor(LIVE_REASONING_PREVIEW_CHARS / 2);
	return `${reasoning.slice(0, sectionLength)}\n\n… live reasoning preview truncated …\n\n${reasoning.slice(-sectionLength)}`;
};

export class ChatExecutionEngine {

	private readonly toolErrMsgs = {
		rejected: 'Tool call was rejected by the user.',
		interrupted: 'Tool call was interrupted by the user.',
		errWhenStringifying: (error: any) => `Tool call succeeded, but there was an error stringifying the output.\n${getErrorMessage(error)}`
	};

	// Pending tool calls from the current turn that are waiting for their approval queue.
	// When LLM returns multiple tool calls in parallel, the first is set as
	// tool_request, the rest are stored here. After Approve/Reject/Skip
	// chatThreadService advances the queue. Only when the queue is empty
	// a new LLM loop is started.
	private readonly _pendingToolCallsByThread = new Map<string, RawToolCallObj[]>();
	private readonly _runningToolInterruptorsByThread = new Map<string, Map<string, () => void>>();
	private readonly _stoppedToolIdsByThread = new Map<string, Set<string>>();
	private readonly _runGenerationByThread = new Map<string, number>();

	private _setRunningToolInterruptor(threadId: string, toolId: string, interruptor: () => void): void {
		const existing = this._runningToolInterruptorsByThread.get(threadId);
		if (existing) {
			existing.set(toolId, interruptor);
			return;
		}
		this._runningToolInterruptorsByThread.set(threadId, new Map([[toolId, interruptor]]));
	}

	private _deleteRunningToolInterruptor(threadId: string, toolId: string): void {
		const existing = this._runningToolInterruptorsByThread.get(threadId);
		if (!existing) return;
		existing.delete(toolId);
		if (!existing.size) this._runningToolInterruptorsByThread.delete(threadId);
	}

	private _currentRunGeneration(threadId: string): number {
		return this._runGenerationByThread.get(threadId) ?? 0;
	}

	private _isRunStopped(threadId: string, runGeneration: number): boolean {
		return this._currentRunGeneration(threadId) !== runGeneration;
	}

	private _markToolStopped(threadId: string, toolId: string): void {
		const existing = this._stoppedToolIdsByThread.get(threadId);
		if (existing) {
			existing.add(toolId);
			return;
		}
		this._stoppedToolIdsByThread.set(threadId, new Set([toolId]));
	}

	private _consumeToolStopped(threadId: string, toolId: string): boolean {
		const existing = this._stoppedToolIdsByThread.get(threadId);
		if (!existing?.delete(toolId)) return false;
		if (!existing.size) this._stoppedToolIdsByThread.delete(threadId);
		return true;
	}

	public stopThread(threadId: string): RawToolCallObj[] {
		this._runGenerationByThread.set(threadId, this._currentRunGeneration(threadId) + 1);
		const pending = this._pendingToolCallsByThread.get(threadId) ?? [];
		this._pendingToolCallsByThread.delete(threadId);

		const interruptors = this._runningToolInterruptorsByThread.get(threadId);
		if (interruptors) {
			for (const [toolId, interruptor] of interruptors) {
				this._markToolStopped(threadId, toolId);
				interruptor();
			}
			this._runningToolInterruptorsByThread.delete(threadId);
		}

		return pending;
	}

	public interruptRunningTools(threadId: string): void {
		const interruptors = this._runningToolInterruptorsByThread.get(threadId);
		if (!interruptors) return;
		for (const [toolId, interruptor] of interruptors) {
			this.skippedToolCallIds.add(toolId);
			interruptor();
		}
		this._runningToolInterruptorsByThread.delete(threadId);
	}

	private _stopPendingToolCalls(threadId: string, access: IThreadStateAccess): void {
		const pending = this._pendingToolCallsByThread.get(threadId) ?? [];
		this._pendingToolCallsByThread.delete(threadId);
		for (const toolCall of pending) {
			this._addToolErrorMessage(threadId, toolCall, this.toolErrMsgs.interrupted, access);
		}
	}

	public clearThreadExecutionState(threadId: string): void {
		this._pendingToolCallsByThread.delete(threadId);
		this._runningToolInterruptorsByThread.delete(threadId);
		this._stoppedToolIdsByThread.delete(threadId);
		this._runGenerationByThread.delete(threadId);
	}

	public clearAllExecutionState(): void {
		this._pendingToolCallsByThread.clear();
		this._runningToolInterruptorsByThread.clear();
		this._stoppedToolIdsByThread.clear();
		this._runGenerationByThread.clear();
	}

	private _isThreadExecutionStopped(threadId: string, runGeneration: number): boolean {
		return this._isRunStopped(threadId, runGeneration);
	}

	private _getDisabledToolNamesSet(): Set<string> {
		const arr = this._settingsService.state.globalSettings.disabledToolNames;
		if (!Array.isArray(arr)) return new Set();
		return new Set(arr.map(v => String(v ?? '').trim()).filter(Boolean));
	}

	private _isToolDisabled(name: string): boolean {
		return this._getDisabledToolNamesSet().has(String(name ?? '').trim());
	}

	private _disabledToolError(toolName: string): string {
		return `Tool "${toolName}" is disabled in Void settings.`;
	}

	private _isAgentSkillDisabled(name: string): boolean {
		const settings = this._settingsService.state.globalSettings;
		if (settings.enableAgentSkills === false) return true;
		const disabled = Array.isArray(settings.disabledAgentSkillNames) ? settings.disabledAgentSkillNames : [];
		return disabled.map(v => String(v ?? '').trim()).includes(String(name ?? '').trim());
	}

	public readonly skippedToolCallIds = new Set<string>();

	constructor(
		@ILLMMessageService private readonly _llmMessageService: ILLMMessageService,
		@IToolsService private readonly _toolsService: IToolsService,
		@IVoidSettingsService private readonly _settingsService: IVoidSettingsService,
		@ILanguageModelToolsService private readonly _lmToolsService: ILanguageModelToolsService,
		@IMetricsService private readonly _metricsService: IMetricsService,
		@IConvertToLLMMessageService private readonly _convertToLLMMessagesService: IConvertToLLMMessageService,
		@IFileService private readonly _fileService: IFileService,
		@IMCPService private readonly _mcpService: IMCPService,
		private readonly _historyCompressor: ChatHistoryCompressor,
		private readonly _toolOutputManager: ChatToolOutputManager
	) { }

	public async runChatAgent(opts: {
		threadId: string,
		modelSelection: ModelSelection | null,
		modelSelectionOptions: ModelSelectionOptions | undefined,
		callThisToolFirst?: ToolMessage<any> & { type: 'tool_request' }
	}, access: IThreadStateAccess) {

		const { threadId, modelSelection, modelSelectionOptions, callThisToolFirst } = opts;
		const runGeneration = this._currentRunGeneration(threadId);

		let interruptedWhenIdle = false;
		const idleInterruptor = Promise.resolve(() => { interruptedWhenIdle = true });
		const isStopped = () => interruptedWhenIdle || this._isThreadExecutionStopped(threadId, runGeneration);

		const gs = this._settingsService.state.globalSettings;
		const chatMode = gs.chatMode;
		const chatRetries = gs.chatRetries;
		const retryDelay = gs.retryDelay;
		const { overridesOfModel } = this._settingsService.state;

		let nMessagesSent = 0;
		let shouldSendAnotherMessage = true;
		let isRunningWhenEnd: IsRunningType = undefined

		const loopDetector = new LLMLoopDetector({
			maxTurnsPerPrompt: gs.loopGuardMaxTurnsPerPrompt,
			maxSameAssistantPrefix: gs.loopGuardMaxSameAssistantPrefix,
			maxSameToolCall: gs.loopGuardMaxSameToolCall,
		});


		if (callThisToolFirst) {
			if (isAToolName(callThisToolFirst.name)) {
				const { interrupted } = await this._runToolCall(threadId, callThisToolFirst.name, callThisToolFirst.id, {
					preapproved: true,
					unvalidatedToolParams: callThisToolFirst.rawParams,
					validatedParams: callThisToolFirst.params
				}, access);

				if (interrupted) {
					if (this._consumeToolStopped(threadId, callThisToolFirst.id)) {
						this._addToolErrorMessage(threadId, {
							id: callThisToolFirst.id,
							name: callThisToolFirst.name,
							rawParams: callThisToolFirst.rawParams,
							isDone: true,
							doneParams: []
						}, this.toolErrMsgs.interrupted, access);
						this._stopPendingToolCalls(threadId, access);
						return;
					}
					if (this.skippedToolCallIds.delete(callThisToolFirst.id)) {

					} else {
						access.setStreamState(threadId, undefined);
						access.addUserCheckpoint(threadId);
						return;
					}
				}
			} else {
				// Dynamic tool (MCP)
				if (isStopped()) {
					this._addToolErrorMessage(threadId, {
						id: callThisToolFirst.id,
						name: callThisToolFirst.name,
						rawParams: callThisToolFirst.rawParams,
						isDone: true,
						doneParams: []
					}, this.toolErrMsgs.interrupted, access);
					return;
				}
				if (this._isToolDisabled(callThisToolFirst.name)) {
					const disabledError = this._disabledToolError(callThisToolFirst.name);
					access.addMessageToThread(threadId, {
						role: 'tool',
						type: 'tool_error',
						params: callThisToolFirst.rawParams as any,
						result: disabledError,
						name: callThisToolFirst.name as any,
						content: disabledError,
						displayContent: disabledError,
						id: callThisToolFirst.id,
						rawParams: callThisToolFirst.rawParams,
					});
				} else {
					access.updateLatestTool(threadId, {
						role: 'tool',
						type: 'running_now',
						params: callThisToolFirst.params as any,
						name: callThisToolFirst.name as any,
						content: 'running...',
						displayContent: 'running...',
						result: null,
						id: callThisToolFirst.id,
						rawParams: callThisToolFirst.rawParams
					});

					const exec = await this._runDynamicToolExec(
						callThisToolFirst.name,
						toJsonObject(callThisToolFirst.rawParams)
					);

					if (isStopped()) {
						access.updateLatestTool(threadId, {
							role: 'tool',
							type: 'tool_error',
							params: callThisToolFirst.params as any,
							result: this.toolErrMsgs.interrupted,
							name: callThisToolFirst.name as any,
							content: this.toolErrMsgs.interrupted,
							displayContent: this.toolErrMsgs.interrupted,
							id: callThisToolFirst.id,
							rawParams: callThisToolFirst.rawParams
						});
						this._stopPendingToolCalls(threadId, access);
						return;
					}

					if (!exec.ok) {
						access.updateLatestTool(threadId, {
							role: 'tool',
							type: 'tool_error',
							params: callThisToolFirst.params as any,
							result: exec.error,
							name: callThisToolFirst.name as any,
							content: exec.error,
							displayContent: exec.error,
							id: callThisToolFirst.id,
							rawParams: callThisToolFirst.rawParams
						});
					} else {
						const { result: processedResult, content, displayContent } =
							await this._toolOutputManager.processToolResult(exec.value, callThisToolFirst.name);

						access.updateLatestTool(threadId, {
							role: 'tool',
							type: 'success',
							params: callThisToolFirst.params as any,
							result: processedResult,
							name: callThisToolFirst.name as any,
							content,
							displayContent: displayContent,
							id: callThisToolFirst.id,
							rawParams: callThisToolFirst.rawParams
						});
					}
				}
			}

		}

		if (isStopped()) {
			this._stopPendingToolCalls(threadId, access);
			return;
		}

		access.setStreamState(threadId, { isRunning: 'idle', interrupt: 'not_needed' });

		// If there are pending tool calls in this turn - do NOT start LLM loop.
		// Wait for user Approve/Reject/Skip on the next tool call from pending.
		// Only when all tool calls are processed, messages can be sent to the provider,
		// otherwise LLM will see tool_request without tool_result and get confused.
		if (this._pendingToolCallsByThread.has(threadId)) {
			access.setStreamState(threadId, { isRunning: 'awaiting_user' });
			return;
		}

		while (shouldSendAnotherMessage) {
			shouldSendAnotherMessage = false;
			isRunningWhenEnd = undefined;
			nMessagesSent += 1;

			access.setStreamState(threadId, { isRunning: 'idle', interrupt: idleInterruptor });


			const baseChatMessages = access.getThreadMessages(threadId);
			let compressionInfoForSentPayload: ThreadHistoryCompressionInfo | undefined;
			let compressedChatMessages: ChatMessage[] | undefined;

			if (nMessagesSent === 1) {
				try {
					const lastUsage = access.getThreadState?.(threadId)?.tokenUsageLastRequest as (LLMTokenUsage | undefined);
					const lastProviderPromptTokens = lastUsage
						? (lastUsage.input + lastUsage.cacheCreation + lastUsage.cacheRead)
						: undefined;

					const compressionResult = await this._historyCompressor.maybeSummarizeHistoryBeforeLLM({
						threadId,
						messages: baseChatMessages,
						modelSelection,
						modelSelectionOptions,
						lastProviderPromptTokens,
					});
					if (compressionResult.summaryText && compressionResult.compressionInfo && compressionResult.compactedMessages?.length) {
						compressedChatMessages = compressionResult.compactedMessages;
						compressionInfoForSentPayload = compressionResult.compressionInfo;
					}
				} catch { /* fail open */ }
			}

			const chatMessages: ChatMessage[] = compressedChatMessages ?? baseChatMessages;
			if (compressionInfoForSentPayload) {
				access.setThreadState(threadId, { historyCompression: compressionInfoForSentPayload });
			}

			const { messages, separateSystemMessage } = await this._convertToLLMMessagesService.prepareLLMChatMessages({
				chatMessages,
				modelSelection,
				chatMode
			});

			await this._patchImagesIntoMessages({ messages, chatMessages, modelSelection });

			if (isStopped()) {
				return;
			}

			let shouldRetryLLM = true;
			let nAttempts = 0;


			while (shouldRetryLLM) {
				shouldRetryLLM = false;
				nAttempts += 1;
				let lastUsageForTurn: LLMTokenUsage | undefined;

				try {
					if (modelSelection) {
						const { providerName, modelName } = modelSelection;
						const caps = getModelCapabilities(providerName as any, modelName, overridesOfModel);
						const reservedFromCaps = caps.reservedOutputTokenSpace ?? 0;

						// If per-model requestParams has max_tokens - this is a more accurate
						// output reserve than caps.reservedOutputTokenSpace. Take the minimum
						// so maxInputTokens is as close as possible to the real provider limit.
						let reservedFromRequestParams = Number.POSITIVE_INFINITY;
						try {
							const customProviders = this._settingsService.state.customProviders || {};
							// Find custom provider config by providerName (supports custom slug)
							const slug = Object.keys(customProviders).find(s => s.toLowerCase() === String(providerName).toLowerCase());
							const cp = slug ? customProviders[slug] : undefined;
							const perModel = (cp?.perModel || {}) as Record<string, any>;
							const cfg = perModel?.[modelName] ?? perModel?.[`${providerName}/${modelName}`];
							const rp = cfg?.requestParams as { mode: 'default' | 'override'; params?: Record<string, any> } | undefined;
							if (rp && (rp.mode === 'default' || rp.mode === 'override') && rp.params && typeof rp.params === 'object') {
								const mt = rp.params.max_tokens ?? rp.params.max_completion_tokens;
								if (typeof mt === 'number' && mt > 0) {
									reservedFromRequestParams = mt;
								}
							}
						} catch { /* ignore */ }

						const reserved = Math.min(reservedFromCaps, reservedFromRequestParams);
						const maxInputTokens = Math.max(0, caps.contextWindow - reserved);
						access.setThreadState(threadId, { tokenUsageLastRequestLimits: { maxInputTokens } });
					}
				} catch { /* noop */ }

				type ResTypes =
					| { type: 'llmDone'; toolCalls?: RawToolCallObj[]; toolCall?: RawToolCallObj; info: { fullText: string; fullReasoning: string; anthropicReasoning: any }; tokenUsage?: LLMTokenUsage }
					| { type: 'llmError'; error?: { message: string; fullError: Error | null } }
					| { type: 'llmAborted' };

				let resMessageIsDonePromise: (res: ResTypes) => void;
				const messageIsDonePromise = new Promise<ResTypes>((res) => { resMessageIsDonePromise = res; });
				let pendingStreamUpdate: Parameters<OnText>[0] | undefined;
				let streamUpdateTimer: ReturnType<typeof setTimeout> | undefined;
				let lastStreamUpdateAt = 0;

				const clearPendingStreamUpdate = () => {
					if (streamUpdateTimer !== undefined) {
						clearTimeout(streamUpdateTimer);
						streamUpdateTimer = undefined;
					}
					pendingStreamUpdate = undefined;
				};

				const publishStreamUpdate = (update: Parameters<OnText>[0]) => {
					if (isStopped()) return;
					lastStreamUpdateAt = Date.now();
					const displayToolCall = update.toolCall ?? update.toolCalls?.[0] ?? null;
					const reasoningPreview = toLiveReasoningPreview(update.fullReasoning);
					access.setStreamState(threadId, {
						isRunning: 'LLM',
						llmInfo: { displayContentSoFar: update.fullText, reasoningSoFar: reasoningPreview, toolCallSoFar: displayToolCall },
						interrupt: Promise.resolve(() => { if (llmCancelToken) this._llmMessageService.abort(llmCancelToken); })
					});
				};

				const queueStreamUpdate = (update: Parameters<OnText>[0]) => {
					pendingStreamUpdate = update;
					const elapsed = Date.now() - lastStreamUpdateAt;
					if (elapsed >= STREAM_UPDATE_INTERVAL_MS && streamUpdateTimer === undefined) {
						pendingStreamUpdate = undefined;
						publishStreamUpdate(update);
						return;
					}
					if (streamUpdateTimer !== undefined) return;
					streamUpdateTimer = setTimeout(() => {
						streamUpdateTimer = undefined;
						const next = pendingStreamUpdate;
						pendingStreamUpdate = undefined;
						if (next) publishStreamUpdate(next);
					}, Math.max(0, STREAM_UPDATE_INTERVAL_MS - elapsed));
				};

				const llmCancelToken = this._llmMessageService.sendLLMMessage({
					messagesType: 'chatMessages',
					chatMode,
					messages: messages,
					modelSelection,
					modelSelectionOptions,
					overridesOfModel,
					logging: { loggingName: `Chat - ${chatMode}`, loggingExtras: { threadId, nMessagesSent, chatMode } },
					separateSystemMessage: separateSystemMessage,
					onText: (update) => {
						if (isStopped()) return;
						if (update.tokenUsage) lastUsageForTurn = update.tokenUsage;
						queueStreamUpdate(update);
					},
					onFinalMessage: async ({ fullText, fullReasoning, toolCalls, toolCall, anthropicReasoning, tokenUsage, }) => {
						clearPendingStreamUpdate();
						if (isStopped()) return;
						if (tokenUsage) lastUsageForTurn = tokenUsage;
						resMessageIsDonePromise({ type: 'llmDone', toolCalls, toolCall, info: { fullText, fullReasoning, anthropicReasoning }, tokenUsage });
					},
					onError: async (error) => {
						clearPendingStreamUpdate();
						if (isStopped()) return;
						resMessageIsDonePromise({ type: 'llmError', error: error });
					},
					onAbort: () => {
						clearPendingStreamUpdate();
						if (isStopped()) return;
						if (lastUsageForTurn) access.accumulateTokenUsage(threadId, lastUsageForTurn);
						resMessageIsDonePromise({ type: 'llmAborted' });
						this._metricsService.capture('Agent Loop Done (Aborted)', { nMessagesSent, chatMode });
					},
				});

				if (!llmCancelToken) {
					access.setStreamState(threadId, { isRunning: undefined, error: { message: 'Unexpected error sending chat message.', fullError: null } });
					break;
				}

				if (isStopped()) {
					this._llmMessageService.abort(llmCancelToken);
					return;
				}
				access.setStreamState(threadId, { isRunning: 'LLM', llmInfo: { displayContentSoFar: '', reasoningSoFar: '', toolCallSoFar: null }, interrupt: Promise.resolve(() => this._llmMessageService.abort(llmCancelToken)) });

				const llmRes = await messageIsDonePromise;

				if (isStopped()) return;
				const currStream = access.getStreamState(threadId);
				if (currStream?.isRunning !== 'LLM') return; // interrupted by new thread

				if (llmRes.type === 'llmAborted') {
					access.setStreamState(threadId, undefined);
					return;
				}
				else if (llmRes.type === 'llmError') {
					if (lastUsageForTurn) access.accumulateTokenUsage(threadId, lastUsageForTurn);

					if (nAttempts < chatRetries) {
						shouldRetryLLM = true;
						access.setStreamState(threadId, { isRunning: 'idle', interrupt: idleInterruptor });
						await timeout(retryDelay);
						if (isStopped()) {
							return;
						}
						continue;
					} else {
						const { error } = llmRes;
						const info = access.getStreamState(threadId).llmInfo;
						access.addMessageToThread(threadId, {
							role: 'assistant',
							displayContent: info.displayContentSoFar,
							reasoning: info.reasoningSoFar,
							anthropicReasoning: null,
							...(lastUsageForTurn ? { tokenUsage: lastUsageForTurn } : {}),
						});
						if (info.toolCallSoFar) access.addMessageToThread(threadId, { role: 'interrupted_streaming_tool', name: info.toolCallSoFar.name });

						access.setStreamState(threadId, { isRunning: undefined, error });
						access.addUserCheckpoint(threadId);
						return;
					}
				}

				// Success
				const { toolCalls, toolCall, info, tokenUsage } = llmRes;
				const effectiveUsage = tokenUsage ?? lastUsageForTurn;
				if (effectiveUsage) access.accumulateTokenUsage(threadId, effectiveUsage);

				access.addMessageToThread(threadId, {
					role: 'assistant',
					displayContent: info.fullText,
					reasoning: info.fullReasoning,
					anthropicReasoning: info.anthropicReasoning,
					...(effectiveUsage ? { tokenUsage: effectiveUsage } : {}),
				});

				// Loop Detection (Assistant)
				const loopAfterAssistant = loopDetector.registerAssistantTurn(info.fullText);
				if (loopAfterAssistant.isLoop) {
					access.setStreamState(threadId, { isRunning: undefined, error: { message: LOOP_DETECTED_MESSAGE, fullError: null } });
					access.addUserCheckpoint(threadId);
					return;
				}

				access.setStreamState(threadId, { isRunning: 'idle', interrupt: 'not_needed' });

				const returnedToolCalls = (toolCalls?.length ? toolCalls : (toolCall ? [toolCall] : []))
					.filter((call): call is RawToolCallObj => !!call?.name);

				if (returnedToolCalls.length) {
					const batchResult = await this._runReturnedToolCalls(threadId, returnedToolCalls, loopDetector, access);
					if (batchResult.loopDetected) {
						access.setStreamState(threadId, { isRunning: undefined, error: { message: LOOP_DETECTED_MESSAGE, fullError: null } });
						access.addUserCheckpoint(threadId);
						return;
					}
					if (batchResult.interrupted) {
						if (!isStopped()) access.setStreamState(threadId, undefined);
						return;
					}
					if (batchResult.awaitingUserApproval) {
						isRunningWhenEnd = 'awaiting_user';
					} else {
						shouldSendAnotherMessage = true;
					}
					access.setStreamState(threadId, { isRunning: 'idle', interrupt: 'not_needed' });
				}
			}
		}

		access.setStreamState(threadId, { isRunning: isRunningWhenEnd });
		if (!isRunningWhenEnd) access.addUserCheckpoint(threadId);
		this._metricsService.capture('Agent Loop Done', { nMessagesSent, chatMode });
	}

	private _addToolErrorMessage(
		threadId: string,
		toolCall: RawToolCallObj,
		error: string,
		access: IThreadStateAccess
	): void {
		access.addMessageToThread(threadId, {
			role: 'tool',
			type: 'tool_error',
			params: toolCall.rawParams as any,
			result: error,
			name: toolCall.name as any,
			content: error,
			displayContent: error,
			id: toolCall.id,
			rawParams: toolCall.rawParams
		});
	}



	private async _runReturnedToolCalls(
		threadId: string,
		toolCalls: readonly RawToolCallObj[],
		loopDetector: LLMLoopDetector,
		access: IThreadStateAccess
	): Promise<{ awaitingUserApproval?: boolean; interrupted?: boolean; loopDetected?: boolean }> {
		const runGeneration = this._currentRunGeneration(threadId);
		const seenWriteTargets = new Set<string>();
		let executedOrErrored = false;
		let pendingParallelToolExecutions: Promise<{ toolCall: RawToolCallObj; awaitingUserApproval?: boolean; interrupted?: boolean }>[] = [];

		const drainPendingParallelToolExecutions = async (): Promise<{ awaitingUserApproval?: boolean; interrupted?: boolean }> => {
			if (!pendingParallelToolExecutions.length) return {};
			const batch = pendingParallelToolExecutions;
			pendingParallelToolExecutions = [];
			const results = await Promise.all(batch);
			const stopped = this._isThreadExecutionStopped(threadId, runGeneration);
			for (const result of results) {
				if (result.interrupted) {
					if (stopped && this._consumeToolStopped(threadId, result.toolCall.id)) {
						executedOrErrored = true;
						continue;
					}
					if (this.skippedToolCallIds.delete(result.toolCall.id)) {
						executedOrErrored = true;
						continue;
					}
					return { interrupted: true };
				}
				if (result.awaitingUserApproval) {
					return { awaitingUserApproval: true };
				}
				executedOrErrored = true;
			}
			if (stopped) {
				this._stopPendingToolCalls(threadId, access);
				return { interrupted: true };
			}
			return {};
		};

		for (let i = 0; i < toolCalls.length; i += 1) {
			if (this._isThreadExecutionStopped(threadId, runGeneration)) {
				this._stopPendingToolCalls(threadId, access);
				for (const remainingToolCall of toolCalls.slice(i)) {
					this._addToolErrorMessage(threadId, remainingToolCall, this.toolErrMsgs.interrupted, access);
				}
				return { interrupted: true };
			}
			const toolCall = toolCalls[i];
			const plan = classifyToolCall({ id: toolCall.id, name: toolCall.name, rawParams: toolCall.rawParams as Record<string, unknown> });

			if (plan.kind !== 'read-only' && plan.kind !== 'read-only-terminal') {
				const pendingResult = await drainPendingParallelToolExecutions();
				if (pendingResult.interrupted || pendingResult.awaitingUserApproval) return pendingResult;
			} else if (plan.kind === 'read-only-terminal' && this._settingsService.state.globalSettings.autoApprove.terminal !== true) {
				const pendingResult = await drainPendingParallelToolExecutions();
				if (pendingResult.interrupted || pendingResult.awaitingUserApproval) return pendingResult;
			}

			if (plan.kind === 'mutating') {
				const target = plan.writeTarget;
				if (!target || seenWriteTargets.has(target)) {
					this._addToolErrorMessage(threadId, toolCall, duplicateWriteTargetError(target), access);
					executedOrErrored = true;
					continue;
				}
				seenWriteTargets.add(target);
			}

			const loopAfterTool = loopDetector.registerToolCall(toolCall.name, toolCall.rawParams);
			if (loopAfterTool.isLoop) {
				const pendingResult = await drainPendingParallelToolExecutions();
				if (pendingResult.interrupted || pendingResult.awaitingUserApproval) return pendingResult;
				return { loopDetected: true };
			}

			if (isAToolName(toolCall.name)) {
				const runBuiltInTool = async () => {
					const { awaitingUserApproval, interrupted } = await this._runToolCall(threadId, toolCall.name, toolCall.id, {
						preapproved: false,
						unvalidatedToolParams: toolCall.rawParams
					}, access);
					return { toolCall, awaitingUserApproval, interrupted };
				};

				if (plan.kind === 'read-only' || (plan.kind === 'read-only-terminal' && this._settingsService.state.globalSettings.autoApprove.terminal === true)) {
					pendingParallelToolExecutions.push(runBuiltInTool());
					continue;
				}

				const { awaitingUserApproval, interrupted } = await runBuiltInTool();

				if (interrupted) {
					if (this._consumeToolStopped(threadId, toolCall.id)) {
						this._addToolErrorMessage(threadId, toolCall, this.toolErrMsgs.interrupted, access);
						for (const remainingToolCall of toolCalls.slice(i + 1)) {
							this._addToolErrorMessage(threadId, remainingToolCall, this.toolErrMsgs.interrupted, access);
						}
						this._stopPendingToolCalls(threadId, access);
						return { interrupted: true };
					}
					if (this.skippedToolCallIds.delete(toolCall.id)) {
						executedOrErrored = true;
						continue;
					}
					return { interrupted: true };
				}

				if (awaitingUserApproval) {
					// Save remaining tool calls to pending buffer. After Approve/Reject/Skip
					// chatThreadService will take the next one from here. Only when all tool calls
					// are processed, a new LLM loop is started.
					const remaining = toolCalls.slice(i + 1);
					if (remaining.length) {
						const existing = this._pendingToolCallsByThread.get(threadId) ?? [];
						this._pendingToolCallsByThread.set(threadId, [...existing, ...remaining]);
					}
					return { awaitingUserApproval: true };
				}

				executedOrErrored = true;
				continue;
			}

			if (this._isToolDisabled(toolCall.name)) {
				this._addToolErrorMessage(threadId, toolCall, this._disabledToolError(toolCall.name), access);
				executedOrErrored = true;
				continue;
			}

			if (!this._settingsService.state.globalSettings.mcpAutoApprove) {
				access.addMessageToThread(threadId, {
					role: 'tool',
					type: 'tool_request',
					content: '(Awaiting user permission...)',
					result: null,
					name: toolCall.name as any,
					params: toolCall.rawParams as any,
					id: toolCall.id,
					rawParams: toolCall.rawParams
				});
				// Save remaining tool calls to pending buffer (see comment above).
				const remaining = toolCalls.slice(i + 1);
				if (remaining.length) {
					const existing = this._pendingToolCallsByThread.get(threadId) ?? [];
					this._pendingToolCallsByThread.set(threadId, [...existing, ...remaining]);
				}
				return { awaitingUserApproval: true };
			}

			access.updateLatestTool(threadId, {
				role: 'tool',
				type: 'running_now',
				name: toolCall.name as any,
				params: toolCall.rawParams as any,
				content: 'running...',
				displayContent: 'running...',
				result: null,
				id: toolCall.id,
				rawParams: toolCall.rawParams
			});

			const exec = await this._runDynamicToolExec(
				toolCall.name,
				toJsonObject(toolCall.rawParams)
			);

			if (this._isThreadExecutionStopped(threadId, runGeneration)) {
				access.updateLatestTool(threadId, {
					role: 'tool',
					type: 'tool_error',
					params: toolCall.rawParams as any,
					result: this.toolErrMsgs.interrupted,
					name: toolCall.name as any,
					content: this.toolErrMsgs.interrupted,
					displayContent: this.toolErrMsgs.interrupted,
					id: toolCall.id,
					rawParams: toolCall.rawParams
				});
				for (const remainingToolCall of toolCalls.slice(i + 1)) {
					this._addToolErrorMessage(threadId, remainingToolCall, this.toolErrMsgs.interrupted, access);
				}
				this._stopPendingToolCalls(threadId, access);
				return { interrupted: true };
			}

			if (!exec.ok) {
				access.updateLatestTool(threadId, {
					role: 'tool',
					type: 'tool_error',
					params: toolCall.rawParams as any,
					result: exec.error,
					name: toolCall.name as any,
					content: exec.error,
					displayContent: exec.error,
					id: toolCall.id,
					rawParams: toolCall.rawParams
				});
			} else {
				const { result: processedResult, content, displayContent } =
					await this._toolOutputManager.processToolResult(exec.value, toolCall.name);

				access.updateLatestTool(threadId, {
					role: 'tool',
					type: 'success',
					params: toolCall.rawParams as any,
					result: processedResult,
					name: toolCall.name as any,
					content,
					displayContent,
					id: toolCall.id,
					rawParams: toolCall.rawParams
				});
			}

			executedOrErrored = true;
		}

		const pendingResult = await drainPendingParallelToolExecutions();
		if (pendingResult.interrupted || pendingResult.awaitingUserApproval) return pendingResult;

		return executedOrErrored ? {} : {};
	}

	private async _runToolCall(
		threadId: string,
		toolName: string,
		toolId: string,
		opts: { preapproved: boolean, unvalidatedToolParams: RawToolParamsObj, validatedParams?: any },
		access: IThreadStateAccess
	): Promise<{ awaitingUserApproval?: boolean, interrupted?: boolean }> {

		let toolParams: any;
		let toolResult: any;

		const isTerminalTool = toolName === 'run_command' || toolName === 'run_persistent_command';

		if (this._isToolDisabled(toolName)) {
			const disabledError = this._disabledToolError(toolName);
			access.addMessageToThread(threadId, {
				role: 'tool',
				type: 'tool_error',
				params: (opts.validatedParams ?? opts.unvalidatedToolParams) as any,
				result: disabledError,
				name: toolName as any,
				content: disabledError,
				displayContent: disabledError,
				id: toolId,
				rawParams: opts.unvalidatedToolParams
			});
			return {};
		}

		if (toolName === 'activate_skill') {
			const requestedName = String(((opts.validatedParams ?? opts.unvalidatedToolParams) as any)?.name ?? '').trim();
			const active = requestedName ? access.getThreadState(threadId)?.activeSkills?.[requestedName] : undefined;
			if (requestedName && active && !this._isAgentSkillDisabled(requestedName)) {
				const contentForModel = formatAgentSkillActivationContent({
					name: requestedName,
					body: '',
					skillFileUri: active.skillFileUri,
					skillDirUri: '',
					resources: [],
					alreadyActive: true,
				});
				access.updateLatestTool(threadId, {
					role: 'tool',
					type: 'success',
					params: (opts.validatedParams ?? { name: requestedName }) as any,
					result: {
						name: requestedName,
						body: '',
						skillFileUri: active.skillFileUri,
						skillDirUri: '',
						resources: [],
						diagnostics: [],
						contentForModel,
						alreadyActive: true,
					} as any,
					name: toolName,
					content: contentForModel,
					displayContent: contentForModel,
					id: toolId,
					rawParams: opts.unvalidatedToolParams,
				});
				return {};
			}
		}

		// 1. Validation & Approval
		if (!opts.preapproved) {
			try {
				if (isAToolName(toolName)) {
					toolParams = this._toolsService.validateParams[toolName](opts.unvalidatedToolParams);
				} else {
					toolParams = opts.unvalidatedToolParams;
				}
			} catch (error) {
				const errorMessage = getErrorMessage(error);
				access.addMessageToThread(threadId, {
					role: 'tool',
					type: 'invalid_params',
					rawParams: opts.unvalidatedToolParams,
					result: null,
					name: toolName as any,
					content: errorMessage,
					id: toolId
				});
				return {};
			}

			if (isAToolName(toolName)) {
				const approvalType = approvalTypeOfToolName[toolName];
				if (approvalType) {
					let autoApprove = this._settingsService.state.globalSettings.autoApprove[approvalType];
					if (approvalType === 'terminal' && toolName === 'run_command') {
						try {
							const cmd = (toolParams as any)?.command ?? String((opts.unvalidatedToolParams as any)?.command ?? '');
							if (isDangerousTerminalCommand(cmd)) autoApprove = false;
						} catch { }
					}
					if (!autoApprove) {
						access.addMessageToThread(threadId, {
							role: 'tool',
							type: 'tool_request',
							content: '(Awaiting user permission...)',
							result: null,
							name: toolName as any,
							params: toolParams,
							id: toolId,
							rawParams: opts.unvalidatedToolParams
						});
						return { awaitingUserApproval: true };
					}
				}
			} else {
				access.addMessageToThread(threadId, {
					role: 'tool',
					type: 'tool_request',
					content: '(Awaiting user permission...)',
					result: null,
					name: toolName as any,
					params: toolParams,
					id: toolId,
					rawParams: opts.unvalidatedToolParams
				});
				return { awaitingUserApproval: true };
			}
		} else {
			toolParams = opts.validatedParams;
		}

		// 2. Execution
		access.updateLatestTool(threadId, {
			role: 'tool',
			type: 'running_now',
			name: toolName as any,
			params: toolParams as any,
			content: '',
			displayContent: '',
			result: null,
			id: toolId,
			rawParams: opts.unvalidatedToolParams
		} as const);

		let interrupted = false;
		let interruptTool: (() => void) | undefined;
		const interruptor = () => { interrupted = true; interruptTool?.(); };
		const interruptorPromise = Promise.resolve(interruptor);
		this._setRunningToolInterruptor(threadId, toolId, interruptor);

		// streamState init
		access.setStreamState(threadId, {
			isRunning: 'tool',
			interrupt: interruptorPromise,
			toolInfo: {
				toolName: isAToolName(toolName) ? toolName : (toolName as any),
				toolParams: toolParams as any,
				id: toolId,
				content: '',
				rawParams: opts.unvalidatedToolParams
			}
		});

		// streaming accumulator
		let streamed = '';
		let pushTimer: any = null;
		let lastPushAt = 0;
		const PUSH_INTERVAL_MS = 80;
		const MAX_KEEP = 200_000;

		const push = (force: boolean) => {
			if (interrupted) return;
			const now = Date.now();
			if (!force && now - lastPushAt < PUSH_INTERVAL_MS) {
				if (!pushTimer) {
					pushTimer = setTimeout(() => {
						pushTimer = null;
						push(true);
					}, PUSH_INTERVAL_MS);
				}
				return;
			}
			lastPushAt = now;

			access.setStreamState(threadId, {
				isRunning: 'tool',
				interrupt: interruptorPromise,
				toolInfo: {
					toolName: isAToolName(toolName) ? toolName : (toolName as any),
					toolParams: toolParams as any,
					id: toolId,
					content: streamed,
					rawParams: opts.unvalidatedToolParams
				}
			});
		};

		// For ephemeral commands show "$ cmd" immediately in stream
		if (toolName === 'run_command') {
			const cmd = String((toolParams as any)?.command ?? '');
			if (cmd) {
				streamed = `$ ${cmd}\n`;
				push(true);
			}
		}

		const onOutput = (chunk: string) => {
			if (interrupted) return;
			if (typeof chunk !== 'string' || !chunk) return;

			streamed += chunk;
			if (streamed.length > MAX_KEEP) {
				streamed = streamed.slice(streamed.length - MAX_KEEP);
			}
			push(false);
		};

		try {
			let result: Promise<any>;

			if (isAToolName(toolName)) {
				// Pass ctx only for terminal tools
				const res = isTerminalTool
					? await (this._toolsService.callTool as any)[toolName](toolParams as any, { onOutput })
					: await this._toolsService.callTool[toolName](toolParams as any);

				result = Promise.resolve(res.result as any);
				interruptTool = res.interruptTool;
			} else {
				result = Promise.resolve({});
			}

			try {
				toolResult = await result;
			} finally {
				this._deleteRunningToolInterruptor(threadId, toolId);
			}

			if (pushTimer) {
				try { clearTimeout(pushTimer); } catch { }
				pushTimer = null;
			}
			push(true);

			if (interrupted || this._consumeToolStopped(threadId, toolId)) return { interrupted: true };
		} catch (error) {
			this._deleteRunningToolInterruptor(threadId, toolId);
			if (interrupted || this._consumeToolStopped(threadId, toolId)) return { interrupted: true };

			const errorMessage = getErrorMessage(error);
			access.updateLatestTool(threadId, {
				role: 'tool',
				type: 'tool_error',
				params: toolParams,
				result: errorMessage,
				name: toolName,
				content: errorMessage,
				displayContent: errorMessage,
				id: toolId,
				rawParams: opts.unvalidatedToolParams
			});
			return {};
		}

		// 3. Stringify & Process Result
		let toolResultStr: string;
		try {
			if (isAToolName(toolName)) {
				toolResultStr = this._toolsService.stringOfResult[toolName](toolParams as any, toolResult as any);
			} else {
				toolResultStr = typeof toolResult === 'string' ? toolResult : JSON.stringify(toolResult);
			}
		} catch (error) {
			const errorMessage = this.toolErrMsgs.errWhenStringifying(error);
			access.updateLatestTool(threadId, {
				role: 'tool',
				type: 'tool_error',
				params: toolParams,
				result: errorMessage,
				name: toolName as any,
				content: errorMessage,
				displayContent: errorMessage,
				id: toolId,
				rawParams: opts.unvalidatedToolParams
			});
			return {};
		}

		let processedResult = toolResult;
		if ((toolName === 'edit_file' || toolName === 'rewrite_file') && toolResult) {
			const resultAny = toolResult as any;
			if (!resultAny.patch_unified && resultAny.preview?.patch_unified) {
				processedResult = { ...toolResult, patch_unified: resultAny.preview.patch_unified };
			}
		}

		const { content, displayContent } = await this._toolOutputManager.processToolResult(toolResultStr, toolName);

		access.updateLatestTool(threadId, {
			role: 'tool',
			type: 'success',
			params: toolParams,
			result: processedResult,
			name: toolName,
			content,
			displayContent,
			id: toolId,
			rawParams: opts.unvalidatedToolParams
		});

		if (toolName === 'activate_skill' && toolResult?.name && toolResult?.skillFileUri) {
			access.markSkillActive(threadId, String(toolResult.name), {
				activatedAt: new Date().toISOString(),
				source: 'tool',
				skillFileUri: String(toolResult.skillFileUri?.toString ? toolResult.skillFileUri.toString() : toolResult.skillFileUri),
			});
		}

		return {};
	}

	private async _runDynamicToolExec(
		name: string,
		args: JsonObject
	): Promise<{ ok: true, value: string | JsonValue } | { ok: false, error: string }> {
		if (this._isToolDisabled(name)) {
			return { ok: false, error: this._disabledToolError(name) };
		}

		try {
			type LmToolShape = { id: string; toolReferenceName?: string; displayName?: string };

			const isToolShape = (v: unknown): v is LmToolShape => {
				if (!isJsonObject(v)) return false;
				return typeof v.id === 'string' && v.id.length > 0;
			};

			// ----------------------------
			// 1) Try execute via ILanguageModelToolsService (settings.json MCP path)
			// ----------------------------
			const toolFromByNameUnknown = this._lmToolsService.getToolByName?.(name) as unknown;
			let tool: LmToolShape | undefined = isToolShape(toolFromByNameUnknown) ? toolFromByNameUnknown : undefined;

			const allToolsUnknown = Array.from(this._lmToolsService.getTools?.() ?? []) as unknown[];
			const allTools: LmToolShape[] = allToolsUnknown.filter(isToolShape);

			if (!tool) {
				for (const t of allTools) {
					if (t.toolReferenceName === name || t.displayName === name) { tool = t; break; }
				}
			}

			// Fallback for prefixed names (e.g. "server__tool")
			if (!tool && name.includes('__')) {
				const baseName = name.split('__').pop();
				if (baseName) {
					for (const t of allTools) {
						if (t.toolReferenceName === baseName || t.displayName === baseName) { tool = t; break; }
					}
				}
			}

			if (tool) {
				const invocation = {
					callId: generateUuid(),
					toolId: tool.id,
					parameters: args ?? {},
					context: undefined,
					skipConfirmation: true,
				};

				const resUnknown = await this._lmToolsService.invokeTool(invocation, async () => 0, CancellationToken.None);

				const tryGetTextParts = (content: unknown): string | null => {
					if (!Array.isArray(content)) return null;
					const texts: string[] = [];
					for (const p of content) {
						if (!p || typeof p !== 'object') continue;
						const kind = (p as { kind?: unknown }).kind;
						const value = (p as { value?: unknown }).value;
						if (kind === 'text' && typeof value === 'string') {
							texts.push(value);
						}
					}
					return texts.length ? texts.join('\n') : null;
				};

				const resObj = isJsonObject(resUnknown) ? (resUnknown as JsonObject) : null;

				const textParts = tryGetTextParts(resObj?.content);
				if (textParts) return { ok: true, value: textParts };

				if (resObj && typeof resObj.toolResultDetails !== 'undefined') return { ok: true, value: resObj.toolResultDetails };
				if (resObj && typeof resObj.toolResultMessage !== 'undefined') return { ok: true, value: resObj.toolResultMessage };

				return { ok: true, value: {} };
			}

			// ----------------------------
			// 2) If not found: try execute via IMCPService (mcp.json path)
			// ----------------------------
			if (name.includes('__')) {
				// Best effort: resolve serverName by searching current MCP state tools
				let resolvedServerName: string | null = null;

				const state = this._mcpService.state?.mcpServerOfName ?? {};
				for (const [serverName, server] of Object.entries(state)) {
					const tools = (server as any)?.tools as Array<{ name: string }> | undefined;
					if (tools?.some(t => t.name === name)) {
						resolvedServerName = serverName;
						break;
					}
				}

				// Fallback: prefix before '__' (works when prefix equals config serverName)
				if (!resolvedServerName) {
					resolvedServerName = name.split('__')[0] || null;
				}

				if (resolvedServerName) {
					const { result } = await this._mcpService.callMCPTool({
						serverName: resolvedServerName,
						toolName: name,
						params: args ?? {},
					});

					const text = this._mcpService.stringifyResult(result);
					return { ok: true, value: text };
				}
			}

			return { ok: false, error: `Unknown dynamic tool: ${name}` };
		} catch (e: unknown) {
			return { ok: false, error: stringifyUnknown(e) };
		}
	}

	private async _patchImagesIntoMessages(opts: { messages: any[]; chatMessages: ChatMessage[]; modelSelection: ModelSelection | null }) {
		const { messages, chatMessages, modelSelection } = opts;
		if (!modelSelection) return;

		const lastUserChat = [...chatMessages].reverse().find(m => m.role === 'user') as (ChatMessage & { attachments?: ChatAttachment[] | null }) | undefined;
		if (!lastUserChat || !lastUserChat.attachments || !lastUserChat.attachments.length) return;

		let lastUserIdx = -1;
		for (let i = messages.length - 1; i >= 0; i -= 1) {
			if (messages[i]?.role === 'user') {
				lastUserIdx = i;
				break;
			}
		}
		if (lastUserIdx === -1) return;

		const lastUser = messages[lastUserIdx];
		const baseContent = typeof lastUser.content === 'string' ? lastUser.content : '';
		const parts: any[] = [];
		const trimmed = baseContent.trim();
		if (trimmed) {
			parts.push({ type: 'text', text: trimmed });
		}

		for (const att of lastUserChat.attachments) {
			try {
				const content = await this._fileService.readFile(att.uri);
				const dataBase64 = (await import('../../../../base/common/buffer.js')).encodeBase64(content.value);
				const mime = (att as any).mimeType || 'image/png';
				const dataUrl = `data:${mime};base64,${dataBase64}`;
				parts.push({ type: 'image_url', image_url: { url: dataUrl } });
			} catch { }
		}
	}
}
