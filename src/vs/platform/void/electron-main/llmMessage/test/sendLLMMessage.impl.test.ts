/*--------------------------------------------------------------------------------------
 *  Copyright 2025 Glass Devtools, Inc. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt for more information.
 *--------------------------------------------------------------------------------------*/

import assert from 'assert';
import { sendChatRouter, runStream, __test as implTestExports } from '../sendLLMMessage.impl.js';
import { setDynamicModelService } from '../../../common/modelInference.js';
import type { OnFinalMessage, OnText } from '../../../common/sendLLMMessageTypes.js';
// eslint-disable-next-line local/code-import-patterns
import { ensureNoDisposablesAreLeakedInTestSuite } from '../../../../../base/test/common/utils.js';

type Delta = {
	content?: string | null | any[]; // <-- allow array-of-parts
	tool_calls?: Array<{
		index?: number;
		type: 'function';
		id?: string;
		function?: { name?: string; arguments?: any };
	}> | null;

	function_call?: { name?: string; arguments?: any } | null;

	[key: string]: any;
};

type Choice = {
	delta?: Delta;
	finish_reason?: 'stop' | 'tool_calls' | 'function_call' | 'length' | null;
};

type Chunk = { choices: [Choice] };

const makeChunk = ({
	content,
	contentParts,
	finish,
	tool,
	legacyFunctionCall,
	reasoningFieldName,
	reasoningDelta,
	reasoningDetails,
}: {
	content?: string | null | any[];
	contentParts?: any[] | null;
	finish?: Choice['finish_reason'];
	tool?: { index?: number; id?: string; name?: string; args?: any } | null;
	legacyFunctionCall?: { name?: string; args?: any } | null;
	reasoningFieldName?: string;
	reasoningDelta?: string | null;
	reasoningDetails?: Array<{ type: string; text: string }> | null;
}): Chunk => {
	const delta: Delta = {};

	// IMPORTANT: contentParts is just a test convenience: we map it to delta.content
	// to simulate providers that stream content as an array-of-parts.
	if (contentParts !== undefined) delta.content = contentParts;
	else if (content !== undefined) delta.content = content;

	if (tool) {
		const tc: any = {
			type: 'function',
			id: tool.id ?? 'call_x',
			function: { name: tool.name, arguments: tool.args ?? '' },
		};
		if (tool.index !== undefined) tc.index = tool.index;
		delta.tool_calls = [tc];
	}

	if (legacyFunctionCall) {
		delta.function_call = {
			name: legacyFunctionCall.name,
			arguments: legacyFunctionCall.args ?? '',
		};
	}

	if (reasoningFieldName && reasoningDelta) {
		delta[reasoningFieldName] = reasoningDelta;
	}

	if (reasoningDetails) {
		delta.reasoning_details = reasoningDetails;
	}

	return { choices: [{ delta, finish_reason: finish ?? null }] };
};

const makeAsyncStream = (chunks: Chunk[], opts?: { abortError?: boolean; delayMs?: number }) => {
	let aborted = false;
	const delayMs = opts?.delayMs ?? 0;

	const controller = {
		abort: () => {
			aborted = true;
		},
	};

	async function* iterator() {
		for (const ch of chunks) {
			if (aborted && opts?.abortError) {
				const err: any = new Error('AbortError');
				err.name = 'AbortError';
				throw err;
			}
			if (delayMs) await new Promise(r => setTimeout(r, delayMs));
			yield ch;
		}
	}

	return {
		controller,
		[Symbol.asyncIterator]: iterator,
	};
};

const makeNonStreamResp = (choice: {
	content?: string;
	tool_calls?: Array<{ id?: string; type: 'function'; function: { name?: string; arguments?: string } }>;
	reasoningFieldName?: string;
	reasoningContent?: string;
	reasoning_details?: Array<{ type: string; text: string }>;
}) => {
	const message: any = { content: choice.content ?? '' };
	if (choice.tool_calls) message.tool_calls = choice.tool_calls;
	if (choice.reasoning_details) message.reasoning_details = choice.reasoning_details;
	if (choice.reasoningFieldName && choice.reasoningContent) {
		message[choice.reasoningFieldName] = choice.reasoningContent;
	}
	return { choices: [{ message }] };
};

const makeFakeOpenAIClient = (response: any) => {
	return {
		chat: {
			completions: {
				create: async (_opts: any) => response,
			},
		},
	} as any;
};

// ---------- capture helpers ----------

const newCaptures = () => {
	const texts: Array<{ fullText: string; fullReasoning: string; toolCall?: any; toolCalls?: any[] }> = [];
	let final: { fullText: string; fullReasoning: string; toolCall?: any; toolCalls?: any[]; anthropicReasoning: any } | null = null;

	const onText: OnText = (p) => texts.push(p);
	const onFinalMessage: OnFinalMessage = (p) => { final = p as any; };

	return {
		texts,
		onText,
		onFinalMessage,
		getFinal: () => final,
		get lastText() {
			return texts[texts.length - 1];
		}
	};
};

const newNotificationCapture = () => {
	const notifications: any[] = [];
	return {
		notifications,
		notificationService: {
			notify: (payload: any) => {
				notifications.push(payload);
			},
		},
	};
};

const newLogCapture = () => {
	const debugLogs: string[] = [];
	return {
		debugLogs,
		logService: {
			debug: (message: string) => {
				debugLogs.push(message);
			},
			warn: (_message: string) => { },
		},
	};
};

const installDebugFetchLoggingForTest = (logService: { debug: (message: string) => void; warn: (message: string) => void }) => {
	implTestExports.installDebugFetchLogging(logService as any);
};

// minimal toolDefsMap (optional)
const toolDefsMap: Map<string, any> = new Map([
	['read_file', { name: 'read_file', params: { uri: {}, start_line: {}, end_line: {} } }],
]);

suite('runStream (OpenAI-compatible)', () => {
	ensureNoDisposablesAreLeakedInTestSuite()
	test('A1: stream only text (content)', async () => {
		const stream = makeAsyncStream([
			makeChunk({ content: 'Hello', finish: null }),
			makeChunk({ content: ' world', finish: 'stop' }),
		]);

		const caps = newCaptures();

		await runStream({
			openai: makeFakeOpenAIClient(stream),
			options: { model: 'o4-mini', messages: [], stream: true } as any,
			onText: caps.onText,
			onFinalMessage: caps.onFinalMessage,
			onError: (e) => assert.fail('onError ' + e.message),
			_setAborter: () => { },
			nameOfReasoningFieldInDelta: undefined,
			providerName: 'openAICompatible' as any,
			toolDefsMap,
		});

		assert.ok(caps.texts.length >= 2);
		assert.strictEqual(caps.lastText.fullText, 'Hello world');
		assert.strictEqual(caps.lastText.fullReasoning, '');
		const fin = caps.getFinal()!;
		assert.strictEqual(fin.fullText, 'Hello world');
		assert.strictEqual(fin.fullReasoning, '');
		assert.strictEqual(fin.toolCall, undefined);
	});

	test('A1b: stream text as array-of-parts (delta.content = [{text}])', async () => {
		const stream = makeAsyncStream([
			makeChunk({ contentParts: [{ type: 'text', text: 'Hello' }], finish: null }),
			makeChunk({ contentParts: [{ type: 'text', text: ' world' }], finish: 'stop' }),
		]);

		const caps = newCaptures();

		await runStream({
			openai: makeFakeOpenAIClient(stream),
			options: { model: 'o4-mini', messages: [], stream: true } as any,
			onText: caps.onText,
			onFinalMessage: caps.onFinalMessage,
			onError: (e) => assert.fail('onError ' + e.message),
			_setAborter: () => { },
			nameOfReasoningFieldInDelta: undefined,
			providerName: 'openAICompatible' as any,
			toolDefsMap,
		});

		const fin = caps.getFinal()!;
		assert.strictEqual(fin.fullText, 'Hello world');
	});

	test('A1c: truncated stream emits notification by default', async () => {
		const stream = makeAsyncStream([
			makeChunk({ content: 'partial answer', finish: 'length' }),
		]);

		const caps = newCaptures();
		const n = newNotificationCapture();

		await runStream({
			openai: makeFakeOpenAIClient(stream),
			options: { model: 'o4-mini', messages: [], stream: true } as any,
			onText: caps.onText,
			onFinalMessage: caps.onFinalMessage,
			onError: (e) => assert.fail('onError ' + e.message),
			_setAborter: () => { },
			nameOfReasoningFieldInDelta: undefined,
			providerName: 'openAICompatible' as any,
			toolDefsMap,
			lengthRetryPolicy: { enabled: false },
			notificationService: n.notificationService as any,
		});

		const fin = caps.getFinal()!;
		assert.strictEqual(fin.fullText, 'partial answer');
		assert.strictEqual(n.notifications.length, 1, 'must notify once on truncation');
		assert.strictEqual(n.notifications[0]?.id, 'void.llm.outputTruncated');
	});

	test('A1d: notifyOnTruncation=false suppresses truncation notification', async () => {
		const stream = makeAsyncStream([
			makeChunk({ content: 'partial answer', finish: 'length' }),
		]);

		const caps = newCaptures();
		const n = newNotificationCapture();

		await runStream({
			openai: makeFakeOpenAIClient(stream),
			options: { model: 'o4-mini', messages: [], stream: true } as any,
			onText: caps.onText,
			onFinalMessage: caps.onFinalMessage,
			onError: (e) => assert.fail('onError ' + e.message),
			_setAborter: () => { },
			nameOfReasoningFieldInDelta: undefined,
			providerName: 'openAICompatible' as any,
			toolDefsMap,
			lengthRetryPolicy: { enabled: false },
			notificationService: n.notificationService as any,
			notifyOnTruncation: false,
		});

		assert.strictEqual(n.notifications.length, 0, 'notification must be suppressed');
	});

	test('A3: reasoning_details overrides reasoning_field', async () => {
		const chunks = [
			makeChunk({ reasoningFieldName: 'reasoning_content', reasoningDelta: 'scratch 1' }),
			makeChunk({ reasoningDetails: [{ type: 'reasoning.text', text: 'real ' }] }),
			makeChunk({ reasoningDetails: [{ type: 'reasoning.text', text: 'reasoning' }], finish: 'stop' }),
		];
		const stream = makeAsyncStream(chunks);

		const caps = newCaptures();

		await runStream({
			openai: makeFakeOpenAIClient(stream),
			options: { model: 'o4-mini', messages: [], stream: true } as any,
			onText: caps.onText,
			onFinalMessage: caps.onFinalMessage,
			onError: (e) => assert.fail('onError ' + e.message),
			_setAborter: () => { },
			nameOfReasoningFieldInDelta: 'reasoning_content',
			providerName: 'openAICompatible' as any,
			toolDefsMap,
		});

		assert.ok(caps.lastText.fullReasoning.endsWith('reasoning'));
		assert.strictEqual(caps.getFinal()!.fullReasoning, 'real reasoning');
	});

	test('B4: openai-style tool_calls (stream, multi-part args)', async () => {
		const partial1 = '{"uri":"/tmp/a.ts", "start_line": 1';
		const partial2 = ', "end_line": 50}';
		const chunks = [
			makeChunk({ tool: { name: 'read_file', id: 'call_1', args: partial1 } }),
			makeChunk({ tool: { name: 'read_file', id: 'call_1', args: partial2 }, finish: 'tool_calls' }),
		];
		const stream = makeAsyncStream(chunks);

		const caps = newCaptures();
		await runStream({
			openai: makeFakeOpenAIClient(stream),
			options: { model: 'o4-mini', messages: [], stream: true } as any,
			onText: caps.onText,
			onFinalMessage: caps.onFinalMessage,
			onError: (e) => assert.fail('onError ' + e.message),
			_setAborter: () => { },
			nameOfReasoningFieldInDelta: undefined,
			providerName: 'openAICompatible' as any,
			toolDefsMap,
		});


		assert.ok(caps.texts.some(t => t.toolCall && t.toolCall.name === 'read_file' && t.toolCall.isDone === false));
		const fin = caps.getFinal()!;
		assert.ok(fin.toolCall, 'final toolCall expected');
		assert.strictEqual(fin.toolCall!.name, 'read_file');
		assert.strictEqual(fin.toolCall!.isDone, true);
	});

	test('B4b: tool_calls without index must still be parsed (no empty response)', async () => {
		const partial1 = '{"uri":"/tmp/a.ts", "start_line": 1';
		const partial2 = ', "end_line": 50}';

		const chunks = [
			// index intentionally omitted
			makeChunk({ tool: { index: undefined, name: 'read_file', id: 'call_1', args: partial1 } }),
			makeChunk({ tool: { index: undefined, name: 'read_file', id: 'call_1', args: partial2 }, finish: 'tool_calls' }),
		];
		const stream = makeAsyncStream(chunks);

		const caps = newCaptures();
		await runStream({
			openai: makeFakeOpenAIClient(stream),
			options: { model: 'o4-mini', messages: [], stream: true } as any,
			onText: caps.onText,
			onFinalMessage: caps.onFinalMessage,
			onError: (e) => assert.fail('onError ' + e.message),
			_setAborter: () => { },
			nameOfReasoningFieldInDelta: undefined,
			providerName: 'openAICompatible' as any,
			toolDefsMap,
		});

		const fin = caps.getFinal()!;
		assert.ok(fin.toolCall, 'final toolCall expected');
		assert.strictEqual(fin.toolCall!.name, 'read_file');
		assert.strictEqual(fin.toolCall!.isDone, true);
	});

	test('B4c: legacy delta.function_call must be supported (stream)', async () => {
		const partial1 = '{"uri":"/tmp/a.ts", "start_line": 1';
		const partial2 = ', "end_line": 2}';

		const chunks = [
			makeChunk({ legacyFunctionCall: { name: 'read_file', args: partial1 } }),
			makeChunk({ legacyFunctionCall: { name: 'read_file', args: partial2 }, finish: 'function_call' }),
		];
		const stream = makeAsyncStream(chunks);

		const caps = newCaptures();
		await runStream({
			openai: makeFakeOpenAIClient(stream),
			options: { model: 'o4-mini', messages: [], stream: true } as any,
			onText: caps.onText,
			onFinalMessage: caps.onFinalMessage,
			onError: (e) => assert.fail('onError ' + e.message),
			_setAborter: () => { },
			nameOfReasoningFieldInDelta: undefined,
			providerName: 'openAICompatible' as any,
			toolDefsMap,
		});

		const fin = caps.getFinal()!;
		assert.ok(fin.toolCall, 'final toolCall expected');
		assert.strictEqual(fin.toolCall!.name, 'read_file');
		assert.strictEqual(fin.toolCall!.isDone, true);
	});

	test('B4d: tool_calls arguments can be an object (not only string)', async () => {
		const chunks = [
			makeChunk({
				tool: {
					// also omit index to cover both at once
					index: undefined,
					id: 'call_2',
					name: 'read_file',
					args: { uri: '/tmp/a.ts', start_line: 1, end_line: 3 }, // <-- object
				},
				finish: 'tool_calls',
			}),
		];
		const stream = makeAsyncStream(chunks);

		const caps = newCaptures();
		await runStream({
			openai: makeFakeOpenAIClient(stream),
			options: { model: 'o4-mini', messages: [], stream: true } as any,
			onText: caps.onText,
			onFinalMessage: caps.onFinalMessage,
			onError: (e) => assert.fail('onError ' + e.message),
			_setAborter: () => { },
			nameOfReasoningFieldInDelta: undefined,
			providerName: 'openAICompatible' as any,
			toolDefsMap,
		});

		const fin = caps.getFinal()!;
		assert.ok(fin.toolCall, 'final toolCall expected');
		assert.strictEqual(fin.toolCall!.name, 'read_file');
		assert.strictEqual(fin.toolCall!.isDone, true);
	});

	test('B4e: stopOnFirstToolCall abort path must still yield final toolCall (AbortError simulation)', async () => {
		// first chunk already contains complete JSON, so runStream may abort immediately
		const chunks = [
			makeChunk({
				tool: { index: undefined, id: 'call_abort', name: 'read_file', args: '{"uri":"/x","start_line":1,"end_line":2}' },
				finish: null,
			}),
			// would be next chunk, but generator will throw AbortError when aborted
			makeChunk({ content: 'SHOULD_NOT_REACH', finish: 'stop' }),
		];

		// abortError=true makes iterator throw AbortError once controller.abort() was called
		const stream = makeAsyncStream(chunks, { abortError: true });

		const caps = newCaptures();
		await runStream({
			openai: makeFakeOpenAIClient(stream),
			options: { model: 'o4-mini', messages: [], stream: true } as any,
			onText: caps.onText,
			onFinalMessage: caps.onFinalMessage,
			onError: (e) => assert.fail('onError ' + e.message),
			_setAborter: () => { },
			nameOfReasoningFieldInDelta: undefined,
			providerName: 'openAICompatible' as any,
			toolDefsMap,
			stopOnFirstToolCall: true,
		});

		const fin = caps.getFinal()!;
		assert.ok(fin.toolCall, 'final toolCall expected');
		assert.strictEqual(fin.toolCall!.name, 'read_file');
	});

	test('B4f: tool_calls with index=1 must still be parsed (no empty response)', async () => {
		const chunks = [
			makeChunk({ tool: { index: 1, id: 'call_i1', name: 'read_file', args: '{"uri":"/x","start_line":1' } }),
			makeChunk({ tool: { index: 1, id: 'call_i1', name: 'read_file', args: ',"end_line":2}' }, finish: 'tool_calls' }),
		];
		const stream = makeAsyncStream(chunks);

		const caps = newCaptures();
		await runStream({
			openai: makeFakeOpenAIClient(stream),
			options: { model: 'o4-mini', messages: [], stream: true } as any,
			onText: caps.onText,
			onFinalMessage: caps.onFinalMessage,
			onError: (e) => assert.fail('onError ' + e.message),
			_setAborter: () => { },
			nameOfReasoningFieldInDelta: undefined,
			providerName: 'openAICompatible' as any,
			toolDefsMap,
		});

		const fin = caps.getFinal()!;
		assert.ok(fin.toolCall);
		assert.strictEqual(fin.toolCall!.name, 'read_file');
		assert.strictEqual(fin.toolCall!.isDone, true);
	});

	test('B4g: openai-style streaming tool_calls preserves multiple indexed calls', async () => {
		const chunks: Chunk[] = [
			{
				choices: [{
					delta: {
						tool_calls: [
							{ index: 0, type: 'function', id: 'call_a', function: { name: 'read_file', arguments: '{"uri":"/a","start_line":1' } },
							{ index: 1, type: 'function', id: 'call_b', function: { name: 'read_file', arguments: '{"uri":"/b","start_line":2' } },
						],
					},
					finish_reason: null,
				}],
			},
			{
				choices: [{
					delta: {
						tool_calls: [
							{ index: 0, type: 'function', function: { arguments: ',"end_line":3}' } },
							{ index: 1, type: 'function', function: { arguments: ',"end_line":4}' } },
						],
					},
					finish_reason: 'tool_calls',
				}],
			},
		];
		const stream = makeAsyncStream(chunks);

		const caps = newCaptures();
		await runStream({
			openai: makeFakeOpenAIClient(stream),
			options: { model: 'o4-mini', messages: [], stream: true } as any,
			onText: caps.onText,
			onFinalMessage: caps.onFinalMessage,
			onError: (e) => assert.fail('onError ' + e.message),
			_setAborter: () => { },
			nameOfReasoningFieldInDelta: undefined,
			providerName: 'openAICompatible' as any,
			toolDefsMap,
			stopOnFirstToolCall: false,
		});

		const fin = caps.getFinal()!;
		assert.strictEqual(fin.toolCalls?.length, 2);
		assert.strictEqual(fin.toolCall, fin.toolCalls![0]);
		assert.deepStrictEqual(fin.toolCalls!.map(t => t.id), ['call_a', 'call_b']);
		assert.deepStrictEqual(fin.toolCalls!.map(t => t.rawParams.uri), ['/a', '/b']);
	});

	test('B4h: openai-style streaming tool_calls preserves multiple unindexed calls with ids', async () => {
		const chunks: Chunk[] = [
			makeChunk({ tool: { id: 'call_a', name: 'read_file', args: '{"uri":"/a","start_line":1' } }),
			makeChunk({ tool: { id: 'call_a', args: ',"end_line":3}' } }),
			makeChunk({ tool: { id: 'call_b', name: 'read_file', args: '{"uri":"/b","start_line":2' } }),
			makeChunk({ tool: { id: 'call_b', args: ',"end_line":4}' }, finish: 'tool_calls' }),
		];
		const stream = makeAsyncStream(chunks);

		const caps = newCaptures();
		await runStream({
			openai: makeFakeOpenAIClient(stream),
			options: { model: 'o4-mini', messages: [], stream: true } as any,
			onText: caps.onText,
			onFinalMessage: caps.onFinalMessage,
			onError: (e) => assert.fail('onError ' + e.message),
			_setAborter: () => { },
			nameOfReasoningFieldInDelta: undefined,
			providerName: 'openAICompatible' as any,
			toolDefsMap,
			stopOnFirstToolCall: false,
		});

		const fin = caps.getFinal()!;
		assert.strictEqual(fin.toolCalls?.length, 2);
		assert.strictEqual(fin.toolCall, fin.toolCalls![0]);
		assert.deepStrictEqual(fin.toolCalls!.map(t => t.id), ['call_a', 'call_b']);
		assert.deepStrictEqual(fin.toolCalls!.map(t => t.rawParams.uri), ['/a', '/b']);
	});

	test('C7: non-stream with content + tool_calls + reasoning_details', async () => {
		const nonStream = makeNonStreamResp({
			content: 'Answer',
			tool_calls: [{ type: 'function', function: { name: 'read_file', arguments: '{"uri":"/x","start_line":1,"end_line":2}' } }],
			reasoning_details: [{ type: 'reasoning.text', text: 'think A' }, { type: 'reasoning.text', text: ' + B' }],
		});

		const caps = newCaptures();
		await runStream({
			openai: makeFakeOpenAIClient(nonStream),
			options: { model: 'o4-mini', messages: [], stream: true } as any,
			onText: caps.onText,
			onFinalMessage: caps.onFinalMessage,
			onError: (e) => assert.fail('onError ' + e.message),
			_setAborter: () => { },
			nameOfReasoningFieldInDelta: 'reasoning_content',
			providerName: 'openAICompatible' as any,
			toolDefsMap,
		});

		assert.strictEqual(caps.texts.length, 0, 'non-stream should not emit progress onText');
		const fin = caps.getFinal()!;
		assert.strictEqual(fin.fullText, 'Answer');
		assert.strictEqual(fin.fullReasoning, 'think A + B');
		assert.ok(fin.toolCall);
		assert.strictEqual(fin.toolCall!.name, 'read_file');
	});

	test('C7b: non-stream returns every message.tool_calls entry', async () => {
		const nonStream = makeNonStreamResp({
			content: 'Answer',
			tool_calls: [
				{ id: 'call_a', type: 'function', function: { name: 'read_file', arguments: '{"uri":"/a","start_line":1,"end_line":2}' } },
				{ id: 'call_b', type: 'function', function: { name: 'read_file', arguments: '{"uri":"/b","start_line":3,"end_line":4}' } },
			],
		});

		const caps = newCaptures();
		await runStream({
			openai: makeFakeOpenAIClient(nonStream),
			options: { model: 'o4-mini', messages: [], stream: true } as any,
			onText: caps.onText,
			onFinalMessage: caps.onFinalMessage,
			onError: (e) => assert.fail('onError ' + e.message),
			_setAborter: () => { },
			nameOfReasoningFieldInDelta: undefined,
			providerName: 'openAICompatible' as any,
			toolDefsMap,
		});

		const fin = caps.getFinal()!;
		assert.strictEqual(fin.toolCalls?.length, 2);
		assert.strictEqual(fin.toolCall, fin.toolCalls![0]);
		assert.deepStrictEqual(fin.toolCalls!.map(t => t.id), ['call_a', 'call_b']);
	});

	test('E13: empty tool name -> no toolCall in final', async () => {
		const nonStream = makeNonStreamResp({
			content: 'ok',
			tool_calls: [{ type: 'function', function: { name: '', arguments: '{}' } }],
		});
		const caps = newCaptures();
		await runStream({
			openai: makeFakeOpenAIClient(nonStream),
			options: { model: 'x', messages: [], stream: true } as any,
			onText: caps.onText,
			onFinalMessage: caps.onFinalMessage,
			onError: (e) => assert.fail('onError ' + e.message),
			_setAborter: () => { },
			nameOfReasoningFieldInDelta: undefined,
			providerName: 'openAICompatible' as any,
		});
		const fin = caps.getFinal()!;
		assert.strictEqual(fin.fullText, 'ok');
		assert.strictEqual(fin.toolCall, undefined);
	});
});

suite('sendLLMMessageToProviderImplementation integrations', () => {
	ensureNoDisposablesAreLeakedInTestSuite()
	setup(() => {
		implTestExports.reset?.();
	});

	teardown(() => {
		implTestExports.reset?.();
	});

	test('OpenAI-compatible debug fetch logging preserves streaming Response body', async () => {
		const log = newLogCapture();
		let providerResponseBody: ReadableStream<Uint8Array> | null = null;
		globalThis.fetch = async (_input: RequestInfo | URL, _init?: RequestInit) => {
			providerResponseBody = new ReadableStream<Uint8Array>({
				start(controller) {
					controller.enqueue(new TextEncoder().encode('data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'));
					controller.close();
				},
			});
			return new Response(providerResponseBody, {
				status: 200,
				statusText: 'OK',
				headers: {
					'content-type': 'text/event-stream',
					'x-api-key': 'response-secret',
				},
			});
		};
		installDebugFetchLoggingForTest(log.logService);

		let fetchResponseBody: ReadableStream<Uint8Array> | null = null;
		class FakeOpenAI {
			constructor(private opts: { fetch: typeof fetch }) { }
			chat = {
				completions: {
					create: async (_opts: any) => {
						const response = await this.opts.fetch('https://openrouter.ai/api/v1/chat/completions', {
							method: 'POST',
							headers: { Authorization: 'Bearer sk-test', 'x-api-key': 'secret-key' },
							body: JSON.stringify({ messages: [{ apiKey: 'secret-in-body' }] }),
						});
						fetchResponseBody = response.body;
						const reader = response.body?.getReader();
						assert.ok(reader, 'streaming body must expose getReader');
						await reader.read();
						reader.releaseLock();
						return makeAsyncStream([
							makeChunk({ content: 'Hello', finish: 'stop' }),
						]);
					},
				},
			};
		}
		implTestExports.setOpenAIModule?.({ default: FakeOpenAI, APIError: class extends Error { } } as any);

		const caps = newCaptures();
		let resolveDone!: () => void;
		const done = new Promise<void>(r => { resolveDone = r; });
		await sendChatRouter({
			messages: [{ role: 'user', content: 'hi' } as any],
			separateSystemMessage: undefined,
			onText: caps.onText,
			onFinalMessage: (p) => { caps.onFinalMessage(p); resolveDone(); },
			onError: (e) => assert.fail('onError: ' + e.message),
			settingsOfProvider: { openrouter: { endpoint: 'https://openrouter.ai/api/v1', apiKey: 'sk-test' } } as any,
			modelSelectionOptions: {} as any,
			overridesOfModel: {} as any,
			modelName: 'openai/gpt-4.1',
			_setAborter: () => { },
			providerName: 'openrouter' as any,
			chatMode: null as any,
			dynamicRequestConfig: {
				endpoint: 'https://openrouter.ai/api/v1',
				apiStyle: 'openai-compatible',
				supportsSystemMessage: 'system-role',
				specialToolFormat: 'openai-style',
				headers: { Authorization: 'Bearer sk-test' },
			} as any,
		});
		await done;

		const fin = caps.getFinal()!;
		assert.strictEqual(fin.fullText, 'Hello');
		assert.strictEqual(fetchResponseBody, providerResponseBody);
		assert.ok(log.debugLogs.some(log => log.includes('HTTP Request POST https://openrouter.ai/api/v1/chat/completions')));
		assert.ok(log.debugLogs.some(log => log.includes('HTTP Response 200 OK for POST https://openrouter.ai/api/v1/chat/completions')));
		assert.ok(log.debugLogs.some(log => log.includes('Bearer ***')));
		assert.ok(log.debugLogs.some(log => log.includes('"x-api-key": "***"')));
		assert.ok(!log.debugLogs.some(log => log.includes('sk-test')));
		assert.ok(!log.debugLogs.some(log => log.includes('secret-key')));
		assert.ok(!log.debugLogs.some(log => log.includes('secret-in-body')));
		assert.ok(!log.debugLogs.some(log => log.includes('response-secret')));
	});

	test('OpenAI-compatible debug fetch logging preserves non-streaming response delivery', async () => {
		const log = newLogCapture();
		globalThis.fetch = async (_input: RequestInfo | URL, _init?: RequestInit) => new Response(JSON.stringify({ choices: [{ message: { content: 'Hello JSON' } }] }), {
			status: 200,
			statusText: 'OK',
			headers: { 'content-type': 'application/json' },
		});
		installDebugFetchLoggingForTest(log.logService);

		class FakeOpenAI {
			constructor(private opts: { fetch: typeof fetch }) { }
			chat = {
				completions: {
					create: async (_opts: any) => {
						const response = await this.opts.fetch('https://openrouter.ai/api/v1/chat/completions', {
							method: 'POST',
							headers: { Authorization: 'Bearer sk-test' },
							body: JSON.stringify({ messages: [{ content: 'hi' }] }),
						});
						return await response.json();
					},
				},
			};
		}
		implTestExports.setOpenAIModule?.({ default: FakeOpenAI, APIError: class extends Error { } } as any);

		const caps = newCaptures();
		let resolveDone!: () => void;
		const done = new Promise<void>(r => { resolveDone = r; });
		await sendChatRouter({
			messages: [{ role: 'user', content: 'hi' } as any],
			separateSystemMessage: undefined,
			onText: caps.onText,
			onFinalMessage: (p) => { caps.onFinalMessage(p); resolveDone(); },
			onError: (e) => assert.fail('onError: ' + e.message),
			settingsOfProvider: { openrouter: { endpoint: 'https://openrouter.ai/api/v1', apiKey: 'sk-test' } } as any,
			modelSelectionOptions: {} as any,
			overridesOfModel: {} as any,
			modelName: 'openai/gpt-4.1',
			_setAborter: () => { },
			providerName: 'openrouter' as any,
			chatMode: null as any,
			dynamicRequestConfig: {
				endpoint: 'https://openrouter.ai/api/v1',
				apiStyle: 'openai-compatible',
				supportsSystemMessage: 'system-role',
				specialToolFormat: 'openai-style',
				headers: { Authorization: 'Bearer sk-test' },
			} as any,
		});
		await done;

		const fin = caps.getFinal()!;
		assert.strictEqual(fin.fullText, 'Hello JSON');
		assert.ok(log.debugLogs.some(log => log.includes('HTTP Request POST https://openrouter.ai/api/v1/chat/completions')));
		assert.ok(log.debugLogs.some(log => log.includes('HTTP Response 200 OK for POST https://openrouter.ai/api/v1/chat/completions')));
		assert.ok(!log.debugLogs.some(log => log.includes('sk-test')));
	});

	test('Anthropic debug fetch logging preserves response delivery', async () => {
		const log = newLogCapture();
		globalThis.fetch = async (_input: RequestInfo | URL, _init?: RequestInit) => new Response(JSON.stringify({ ok: true }), {
			status: 200,
			statusText: 'OK',
			headers: { 'content-type': 'application/json' },
		});
		installDebugFetchLoggingForTest(log.logService);

		class FakeAnthropic {
			static APIError = class extends Error { status = 401 };
			constructor(private opts: { fetch: typeof fetch }) { }
			messages = {
				stream: (_args: any) => {
					const handlers: Record<string, Function[]> = {};
					const api = {
						on: (event: string, cb: Function) => { (handlers[event] ||= []).push(cb); },
						controller: { abort() { } },
					};
					void this.opts.fetch('https://api.anthropic.com/v1/messages', {
						method: 'POST',
						headers: { 'x-api-key': 'anthropic-secret', Authorization: 'Bearer leaked' },
						body: JSON.stringify({ apiKey: 'secret-in-body' }),
					});
					queueMicrotask(() => {
						handlers['streamEvent']?.forEach(fn => fn({ type: 'content_block_start', content_block: { type: 'text', text: 'Anthropic ok' } }));
						handlers['finalMessage']?.forEach(fn => fn({ content: [{ type: 'text', text: 'Anthropic ok' }] }));
					});
					return api;
				},
			};
		}
		implTestExports.setAnthropicModule?.(FakeAnthropic as any);

		const caps = newCaptures();
		let resolveDone!: () => void;
		const done = new Promise<void>(r => { resolveDone = r; });
		await sendChatRouter({
			messages: [{ role: 'user', content: 'hi' } as any],
			providerName: 'anthropic' as any,
			onText: caps.onText,
			onFinalMessage: (p) => { caps.onFinalMessage(p); resolveDone(); },
			onError: (e) => assert.fail('onError: ' + e.message),
			settingsOfProvider: { anthropic: { apiKey: 'anthropic-secret' } } as any,
			modelSelectionOptions: {} as any,
			overridesOfModel: {} as any,
			modelName: 'claude-3',
			_setAborter: () => { },
			separateSystemMessage: undefined,
			chatMode: null as any,
			additionalTools: undefined,
		});
		await done;

		const fin = caps.getFinal()!;
		assert.strictEqual(fin.fullText, 'Anthropic ok');
		assert.ok(log.debugLogs.some(log => log.includes('HTTP Request POST https://api.anthropic.com/v1/messages')));
		assert.ok(log.debugLogs.some(log => log.includes('HTTP Response 200 OK for POST https://api.anthropic.com/v1/messages')));
		assert.ok(log.debugLogs.some(log => log.includes('Bearer ***')));
		assert.ok(log.debugLogs.some(log => log.includes('"x-api-key": "***"')));
		assert.ok(!log.debugLogs.some(log => log.includes('anthropic-secret')));
		assert.ok(!log.debugLogs.some(log => log.includes('secret-in-body')));
	});

	test('Anthropic: streams text/thinking/tool_use, produces final with toolCall', async () => {
		// Fake Anthropic module
		class FakeAnthropic {
			static APIError = class extends Error { status = 401 };
			constructor(_opts: any) { }
			messages = {
				stream: (_args: any) => {
					const handlers: Record<string, Function[]> = {};
					const api = {
						on: (event: string, cb: Function) => {
							(handlers[event] ||= []).push(cb);
						},
						controller: { abort() { /* no-op for test */ } },
					};
					// schedule events
					queueMicrotask(() => {
						// text block start
						handlers['streamEvent']?.forEach(fn => fn({ type: 'content_block_start', content_block: { type: 'text', text: 'Hi' } }));
						// thinking block start
						handlers['streamEvent']?.forEach(fn => fn({ type: 'content_block_start', content_block: { type: 'thinking', thinking: 'thoughts' } }));
						// tool use start (name only)
						handlers['streamEvent']?.forEach(fn => fn({ type: 'content_block_start', content_block: { type: 'tool_use', name: 'read_file', id: 'tool_1' } }));
						// tool args delta
						handlers['streamEvent']?.forEach(fn => fn({ type: 'content_block_delta', delta: { type: 'input_json_delta', partial_json: '{"uri":"/a","start_line":1,"end_line":2}' } }));
						// finalMessage
						const final = {
							content: [
								{ type: 'thinking', thinking: 'final-think' },
								{ type: 'tool_use', id: 'tool_1', name: 'read_file', input: { uri: '/a', start_line: 1, end_line: 2 } },
							],
						};
						handlers['finalMessage']?.forEach(fn => fn(final));
					});
					return api;
				}
			}
		}
		implTestExports.setAnthropicModule?.(FakeAnthropic as any);

		const caps = newCaptures();
		let resolveDone: () => void;
		const done = new Promise<void>(r => { resolveDone = r; });
		const onFinal: OnFinalMessage = (p) => { caps.onFinalMessage(p); resolveDone!(); };
		const onText: OnText = caps.onText;
		const onError = (e: any) => assert.fail('onError: ' + e.message);

		await sendChatRouter({
			messages: [{ role: 'user', content: 'hi' } as any],
			providerName: 'anthropic' as any,
			onText, onFinalMessage: onFinal, onError,
			settingsOfProvider: { anthropic: { apiKey: 'k' } } as any,
			modelSelectionOptions: {} as any,
			overridesOfModel: {} as any,
			modelName: 'claude-3',
			_setAborter: () => { },
			separateSystemMessage: undefined,
			chatMode: null as any,
			additionalTools: undefined,
		});

		await done;

		assert.ok(caps.texts.length > 0);
		const fin = caps.getFinal()!;
		assert.ok(fin, 'final expected');
		assert.ok(fin.toolCall, 'final toolCall expected');
		assert.strictEqual(fin.toolCall.name, 'read_file');
		assert.strictEqual(fin.toolCall.isDone, true);
	});

	test('Anthropic: finalMessage preserves multiple tool_use blocks', async () => {
		class FakeAnthropic {
			static APIError = class extends Error { status = 401 };
			constructor(_opts: any) { }
			messages = {
				stream: (_args: any) => {
					const handlers: Record<string, Function[]> = {};
					const api = {
						on: (event: string, cb: Function) => { (handlers[event] ||= []).push(cb); },
						controller: { abort() { } },
					};
					queueMicrotask(() => {
						handlers['finalMessage']?.forEach(fn => fn({
							content: [
								{ type: 'tool_use', id: 'tool_1', name: 'read_file', input: { uri: '/a', start_line: 1, end_line: 2 } },
								{ type: 'tool_use', id: 'tool_2', name: 'read_file', input: { uri: '/b', start_line: 3, end_line: 4 } },
							],
						}));
					});
					return api;
				}
			}
		}
		implTestExports.setAnthropicModule?.(FakeAnthropic as any);

		const caps = newCaptures();
		let resolveDone!: () => void;
		const done = new Promise<void>(r => { resolveDone = r; });
		const onFinal: OnFinalMessage = (p) => { caps.onFinalMessage(p); resolveDone(); };

		await sendChatRouter({
			messages: [{ role: 'user', content: 'hi' } as any],
			providerName: 'anthropic' as any,
			onText: caps.onText, onFinalMessage: onFinal, onError: (e) => assert.fail('onError: ' + e.message),
			settingsOfProvider: { anthropic: { apiKey: 'k' } } as any,
			modelSelectionOptions: {} as any,
			overridesOfModel: {} as any,
			modelName: 'claude-3',
			_setAborter: () => { },
			separateSystemMessage: undefined,
			chatMode: null as any,
			additionalTools: undefined,
		});

		await done;

		const fin = caps.getFinal()!;
		assert.strictEqual(fin.toolCalls?.length, 2);
		assert.deepStrictEqual(fin.toolCalls!.map(t => t.id), ['tool_1', 'tool_2']);
		assert.strictEqual(fin.toolCall, fin.toolCalls![0]);
	});

	test('Gemini debug fetch logging preserves response delivery', async () => {
		const log = newLogCapture();
		globalThis.fetch = async (_input: RequestInfo | URL, _init?: RequestInit) => new Response(JSON.stringify({ ok: true }), {
			status: 200,
			statusText: 'OK',
			headers: { 'content-type': 'application/json' },
		});
		installDebugFetchLoggingForTest(log.logService);

		class FakeStream {
			private chunks: any[];
			constructor(chunks: any[]) { this.chunks = chunks; }
			async *[Symbol.asyncIterator]() { for (const ch of this.chunks) yield ch; }
			return(_x: any) { }
		}
		class FakeGoogleGenAI {
			constructor(_opts: any) { }
			models = {
				generateContentStream: (_args: any) => {
					void globalThis.fetch('https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5:streamGenerateContent?key=gemini-secret', {
						method: 'POST',
						headers: { 'x-goog-api-key': 'gemini-secret' },
						body: JSON.stringify({ apiKey: 'secret-in-body' }),
					});
					return Promise.resolve(new FakeStream([{ text: 'Gemini ok' }]));
				},
			};
		}
		implTestExports.setGoogleGenAIModule?.({ GoogleGenAI: FakeGoogleGenAI } as any);

		const caps = newCaptures();
		let resolveDone!: () => void;
		const done = new Promise<void>(r => { resolveDone = r; });
		await sendChatRouter({
			messages: [{ role: 'user', parts: [{ text: 'hi' }] } as any],
			separateSystemMessage: undefined,
			onText: caps.onText,
			onFinalMessage: (p) => { caps.onFinalMessage(p); resolveDone(); },
			onError: (e) => assert.fail('onError: ' + e.message),
			settingsOfProvider: { gemini: { apiKey: 'gemini-secret' } } as any,
			overridesOfModel: {} as any,
			modelName: 'gemini-1.5',
			_setAborter: () => { },
			providerName: 'gemini' as any,
			modelSelectionOptions: {} as any,
			chatMode: null as any,
			additionalTools: undefined,
		});
		await done;

		const fin = caps.getFinal()!;
		assert.strictEqual(fin.fullText, 'Gemini ok');
		assert.ok(log.debugLogs.some(log => log.includes('HTTP Request POST https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5:streamGenerateContent?key=***')));
		assert.ok(log.debugLogs.some(log => log.includes('HTTP Response 200 OK for POST https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5:streamGenerateContent?key=***')));
		assert.ok(log.debugLogs.some(log => log.includes('"x-goog-api-key": "***"')));
		assert.ok(!log.debugLogs.some(log => log.includes('gemini-secret')));
		assert.ok(!log.debugLogs.some(log => log.includes('secret-in-body')));
	});

	test('Gemini: generateContentStream yields text and functionCalls, final returns toolCall', async () => {
		// Fake GoogleGenAI module
		class FakeStream {
			private chunks: any[];
			constructor(chunks: any[]) { this.chunks = chunks; }
			async *[Symbol.asyncIterator]() {
				for (const ch of this.chunks) yield ch;
			}
			return(_x: any) { /* allow abort */ }
		}
		class FakeGoogleGenAI {
			constructor(_opts: any) { }
			models = {
				generateContentStream: (_args: any) => {
					const chunks = [
						{ text: 'Hello ' },
						{ text: 'world' },
						{
							functionCalls: [{ name: 'read_file', args: { uri: '/b', start_line: 1, end_line: 3 }, id: 'g1' }]
						},
					];
					return Promise.resolve(new FakeStream(chunks));
				}
			}
		}
		implTestExports.setGoogleGenAIModule?.({ GoogleGenAI: FakeGoogleGenAI } as any);

		const caps = newCaptures();
		let resolveDone: () => void;
		const done = new Promise<void>(r => { resolveDone = r; });
		const onFinal: OnFinalMessage = (p) => { caps.onFinalMessage(p); resolveDone!(); };
		const onText: OnText = caps.onText;
		const onError = (e: any) => assert.fail('onError: ' + e.message);

		await sendChatRouter({
			messages: [{ role: 'user', parts: [{ text: 'hi' }] } as any],
			separateSystemMessage: undefined,
			onText, onFinalMessage: onFinal, onError,
			settingsOfProvider: { gemini: { apiKey: 'k' } } as any,
			overridesOfModel: {} as any,
			modelName: 'gemini-1.5',
			_setAborter: () => { },
			providerName: 'gemini' as any,
			modelSelectionOptions: {} as any,
			chatMode: null as any,
			additionalTools: undefined,
		});

		await done;

		assert.ok(caps.texts.length >= 2, 'should emit text progress');
		const fin = caps.getFinal()!;
		assert.ok(fin, 'final expected');
		assert.strictEqual(fin.fullText, 'Hello world');
		assert.ok(fin.toolCall);
		assert.strictEqual(fin.toolCall.name, 'read_file');
		assert.strictEqual(fin.toolCall.isDone, true);
	});

	test('Gemini: preserves multiple functionCalls', async () => {
		class FakeStream {
			private chunks: any[];
			constructor(chunks: any[]) { this.chunks = chunks; }
			async *[Symbol.asyncIterator]() { for (const ch of this.chunks) yield ch; }
			return(_x: any) { }
		}
		class FakeGoogleGenAI {
			constructor(_opts: any) { }
			models = {
				generateContentStream: (_args: any) => Promise.resolve(new FakeStream([
					{
						functionCalls: [
							{ name: 'read_file', args: { uri: '/a', start_line: 1, end_line: 2 }, id: 'g1' },
							{ name: 'read_file', args: { uri: '/b', start_line: 3, end_line: 4 }, id: 'g2' },
						]
					},
				]))
			}
		}
		implTestExports.setGoogleGenAIModule?.({ GoogleGenAI: FakeGoogleGenAI } as any);

		const caps = newCaptures();
		let resolveDone!: () => void;
		const done = new Promise<void>(r => { resolveDone = r; });
		const onFinal: OnFinalMessage = (p) => { caps.onFinalMessage(p); resolveDone(); };

		await sendChatRouter({
			messages: [{ role: 'user', parts: [{ text: 'hi' }] } as any],
			separateSystemMessage: undefined,
			onText: caps.onText,
			onFinalMessage: onFinal,
			onError: (e) => assert.fail('onError: ' + e.message),
			settingsOfProvider: { gemini: { apiKey: 'k' } } as any,
			overridesOfModel: {} as any,
			modelName: 'gemini-1.5',
			_setAborter: () => { },
			providerName: 'gemini' as any,
			modelSelectionOptions: {} as any,
			chatMode: null as any,
			additionalTools: undefined,
		});

		await done;

		const fin = caps.getFinal()!;
		assert.strictEqual(fin.toolCalls?.length, 2);
		assert.deepStrictEqual(fin.toolCalls!.map(t => t.id), ['g1', 'g2']);
		assert.strictEqual(fin.toolCall, fin.toolCalls![0]);
	});

	test('stream only reasoning_field (no details)', async () => {
		const chunks = [
			makeChunk({ reasoningFieldName: 'reasoning_content', reasoningDelta: 'foo' }),
			makeChunk({ reasoningFieldName: 'reasoning_content', reasoningDelta: 'bar', finish: 'stop' }),
		];
		const stream = makeAsyncStream(chunks);

		const caps = newCaptures();
		await runStream({
			openai: makeFakeOpenAIClient(stream),
			options: { model: 'x', messages: [], stream: true } as any,
			onText: caps.onText,
			onFinalMessage: caps.onFinalMessage,
			onError: (e) => assert.fail('onError ' + e.message),
			_setAborter: () => { },
			nameOfReasoningFieldInDelta: 'reasoning_content',
			providerName: 'openAICompatible' as any,
		});

		const fin = caps.getFinal()!;
		assert.strictEqual(fin.fullReasoning, 'foobar');
		assert.strictEqual(fin.fullText, '');
	});

	test('stream content + reasoning_field together', async () => {
		const chunks = [
			makeChunk({ content: 'X', reasoningFieldName: 'reasoning_content', reasoningDelta: 'r1' }),
			makeChunk({ content: 'Y', reasoningFieldName: 'reasoning_content', reasoningDelta: 'r2', finish: 'stop' }),
		];
		const stream = makeAsyncStream(chunks);

		const caps = newCaptures();
		await runStream({
			openai: makeFakeOpenAIClient(stream),
			options: { model: 'x', messages: [], stream: true } as any,
			onText: caps.onText,
			onFinalMessage: caps.onFinalMessage,
			onError: (e) => assert.fail('onError ' + e.message),
			_setAborter: () => { },
			nameOfReasoningFieldInDelta: 'reasoning_content',
			providerName: 'openAICompatible' as any,
		});

		const fin = caps.getFinal()!;
		assert.strictEqual(fin.fullText, 'XY');
		assert.strictEqual(fin.fullReasoning, 'r1r2');
	});

	test('Anthropic: includes redacted_thinking along with thinking and tool_use in final', async () => {
		class FakeAnthropic {
			static APIError = class extends Error { status = 401 };
			constructor(_opts: any) { }
			messages = {
				stream: (_args: any) => {
					const handlers: Record<string, Function[]> = {};
					const api = {
						on: (event: string, cb: Function) => { (handlers[event] ||= []).push(cb); },
						controller: { abort() { } },
					};
					queueMicrotask(() => {
						// text + thinking + tool_use start
						handlers['streamEvent']?.forEach(fn => fn({ type: 'content_block_start', content_block: { type: 'text', text: 'Hi' } }));
						handlers['streamEvent']?.forEach(fn => fn({ type: 'content_block_start', content_block: { type: 'thinking', thinking: 'scratch' } }));
						handlers['streamEvent']?.forEach(fn => fn({ type: 'content_block_start', content_block: { type: 'tool_use', name: 'read_file', id: 'tool_1' } }));

						const final = {
							content: [
								{ type: 'thinking', thinking: 'visible-think' },
								{ type: 'redacted_thinking', data: { hidden: true } },
								{ type: 'tool_use', id: 'tool_1', name: 'read_file', input: { uri: '/a', start_line: 1, end_line: 2 } },
							],
						};
						handlers['finalMessage']?.forEach(fn => fn(final));
					});
					return api;
				}
			}
		}
		implTestExports.setAnthropicModule?.(FakeAnthropic as any);

		const caps = newCaptures();
		let resolveDone!: () => void;
		const done = new Promise<void>(r => { resolveDone = r; });
		const onFinal: OnFinalMessage = (p) => { caps.onFinalMessage(p); resolveDone(); };

		await sendChatRouter({
			messages: [{ role: 'user', content: 'hi' } as any],
			providerName: 'anthropic' as any,
			onText: caps.onText,
			onFinalMessage: onFinal,
			onError: (e) => assert.fail('onError: ' + e.message),
			settingsOfProvider: { anthropic: { apiKey: 'k' } } as any,
			modelSelectionOptions: {} as any,
			overridesOfModel: {} as any,
			modelName: 'claude-3',
			_setAborter: () => { },
			separateSystemMessage: undefined,
			chatMode: null as any,
			additionalTools: undefined,
		});

		await done;

		const fin = caps.getFinal()!;
		assert.ok(Array.isArray(fin.anthropicReasoning), 'anthropicReasoning should be array');
		assert.ok(fin.anthropicReasoning.some((x: any) => x.type === 'thinking'));
		assert.ok(fin.anthropicReasoning.some((x: any) => x.type === 'redacted_thinking'));
		assert.ok(fin.toolCall, 'toolCall expected');
		assert.strictEqual(fin.toolCall.name, 'read_file');
		assert.strictEqual(fin.toolCall.isDone, true);
	});

	test('OpenRouter deepseek :free respects dynamicRequestConfig.specialToolFormat="disabled" (no tools sent)', async () => {




		const fakeDynamicService: any = {
			getDynamicCapabilities(modelName: string) {
				if (modelName === 'deepseek/deepseek-r1-0528:free') {
					return {
						contextWindow: 4096,
						reservedOutputTokenSpace: 4096,
						supportsSystemMessage: 'system-role',
						specialToolFormat: 'openai-style',
						supportsFIM: false,
						reasoningCapabilities: false,
						cost: { input: 0, output: 0 },
					};
				}
				return null;
			},
			getAllDynamicCapabilities() {
				return {};
			}
		};
		setDynamicModelService(fakeDynamicService);

		let capturedOptions: any = null;
		class FakeOpenAI {
			chat = {
				completions: {
					create: async (opts: any) => {
						capturedOptions = opts;

						return {
							choices: [{
								message: { content: 'ok', tool_calls: undefined }
							}]
						};
					},
				},
			};
		}
		implTestExports.setOpenAIModule?.({ default: FakeOpenAI, APIError: class extends Error { } } as any);

		const caps = newCaptures();
		let resolveDone!: () => void;
		const done = new Promise<void>(r => { resolveDone = r; });
		const onFinal: OnFinalMessage = (p) => { caps.onFinalMessage(p); resolveDone(); };

		await sendChatRouter({
			messages: [{ role: 'user', content: 'hi' } as any],
			separateSystemMessage: undefined,
			onText: caps.onText,
			onFinalMessage: onFinal,
			onError: (e) => assert.fail('onError: ' + e.message),
			settingsOfProvider: {
				openrouter: {
					endpoint: 'https://openrouter.ai/api/v1',
					apiKey: 'sk-test',
					apiStyle: 'openai-compatible',
				}
			} as any,
			modelSelectionOptions: {} as any,
			overridesOfModel: {} as any,
			modelName: 'deepseek/deepseek-r1-0528:free',
			_setAborter: () => { },
			providerName: 'openrouter' as any,
			chatMode: null as any,
			additionalTools: [
				{ name: 'read_file', description: 'd', params: { uri: {}, start_line: {}, end_line: {} } }
			] as any,
			dynamicRequestConfig: {
				endpoint: 'https://openrouter.ai/api/v1',
				apiStyle: 'openai-compatible',
				supportsSystemMessage: false as any,
				specialToolFormat: 'disabled' as any,
				headers: { Authorization: 'Bearer sk-test', Accept: 'application/json' },
			} as any,
			requestParams: undefined,
		});

		await done;

		assert.ok(capturedOptions, 'OpenAI.create should have been called');



		assert.strictEqual(
			capturedOptions.tools,
			undefined,
			'options.tools must be omitted when specialToolFormat is disabled for the selected model'
		);
	});

	test('OpenAI-compatible payload sends parallel_tool_calls=false when supported native tools are present', async () => {
		setDynamicModelService({
			getDynamicCapabilities(_modelName: string) {
				return {
					contextWindow: 4096,
					reservedOutputTokenSpace: 4096,
					supportsSystemMessage: 'system-role',
					specialToolFormat: 'openai-style',
					supportsFIM: false,
					supportsParallelToolCalls: true,
					reasoningCapabilities: false,
					cost: { input: 0, output: 0 },
				};
			},
			getAllDynamicCapabilities() { return {}; },
		} as any);

		let capturedOptions: any = null;
		class FakeOpenAI {
			chat = {
				completions: {
					create: async (opts: any) => {
						capturedOptions = opts;
						return { choices: [{ message: { content: 'ok' } }] };
					}
				}
			};
		}
		implTestExports.setOpenAIModule?.({ default: FakeOpenAI, APIError: class extends Error { } } as any);

		const caps = newCaptures();
		let resolveDone!: () => void;
		const done = new Promise<void>(r => { resolveDone = r; });

		await sendChatRouter({
			messages: [{ role: 'user', content: 'hi' } as any],
			separateSystemMessage: undefined,
			onText: caps.onText,
			onFinalMessage: (p) => { caps.onFinalMessage(p); resolveDone(); },
			onError: (e) => assert.fail('onError: ' + e.message),
			settingsOfProvider: { openrouter: { endpoint: 'https://openrouter.ai/api/v1', apiKey: 'sk-test' } } as any,
			modelSelectionOptions: {} as any,
			overridesOfModel: {} as any,
			modelName: 'anthropic/claude-3.5',
			_setAborter: () => { },
			providerName: 'openrouter' as any,
			chatMode: 'agent' as any,
			dynamicRequestConfig: {
				endpoint: 'https://openrouter.ai/api/v1',
				apiStyle: 'openai-compatible',
				supportsSystemMessage: 'system-role',
				specialToolFormat: 'openai-style',
				headers: { Authorization: 'Bearer sk-test' },
				parallelToolCalls: { supported: true, mode: 'safe-disabled' },
			} as any,
		});

		await done;

		assert.ok(Array.isArray(capturedOptions.tools), 'native tools should be present');
		assert.strictEqual(capturedOptions.parallel_tool_calls, false);
	});

	test('OpenAI-compatible enabled parallel_tool_calls keeps stream open for multiple tool calls', async () => {
		let capturedOptions: any = null;
		let abortCount = 0;
		const chunks = [
			makeChunk({ tool: { index: 0, id: 'call_a', name: 'read_file', args: '{"uri":"/a"}' }, finish: null }),
			makeChunk({ tool: { index: 1, id: 'call_b', name: 'read_file', args: '{"uri":"/b"}' }, finish: 'tool_calls' }),
		];
		class FakeOpenAI {
			chat = {
				completions: {
					create: async (opts: any) => {
						capturedOptions = opts;
						let aborted = false;
						return {
							controller: {
								abort: () => {
									aborted = true;
									abortCount++;
								},
							},
							async *[Symbol.asyncIterator]() {
								for (const chunk of chunks) {
									if (aborted) break;
									yield chunk;
								}
							},
						};
					}
				}
			};
		}
		implTestExports.setOpenAIModule?.({ default: FakeOpenAI, APIError: class extends Error { } } as any);

		const caps = newCaptures();
		let resolveDone!: () => void;
		const done = new Promise<void>(r => { resolveDone = r; });

		await sendChatRouter({
			messages: [{ role: 'user', content: 'hi' } as any],
			separateSystemMessage: undefined,
			onText: caps.onText,
			onFinalMessage: (p) => { caps.onFinalMessage(p); resolveDone(); },
			onError: (e) => assert.fail('onError: ' + e.message),
			settingsOfProvider: { openrouter: { endpoint: 'https://openrouter.ai/api/v1', apiKey: 'sk-test' } } as any,
			modelSelectionOptions: {} as any,
			overridesOfModel: {} as any,
			modelName: 'openai/gpt-4.1',
			_setAborter: () => { },
			providerName: 'openrouter' as any,
			chatMode: 'agent' as any,
			dynamicRequestConfig: {
				endpoint: 'https://openrouter.ai/api/v1',
				apiStyle: 'openai-compatible',
				supportsSystemMessage: 'system-role',
				specialToolFormat: 'openai-style',
				headers: { Authorization: 'Bearer sk-test' },
				parallelToolCalls: { supported: true, mode: 'enabled' },
			} as any,
		});

		await done;

		const fin = caps.getFinal()!;
		assert.ok(Array.isArray(capturedOptions.tools), 'native tools should be present');
		assert.strictEqual(capturedOptions.parallel_tool_calls, true);
		assert.strictEqual(abortCount, 0);
		assert.strictEqual(fin.toolCalls?.length, 2);
		assert.deepStrictEqual(fin.toolCalls!.map(t => t.id), ['call_a', 'call_b']);
	});

	test('OpenAI-compatible payload omits parallel_tool_calls for unsupported, XML, and no-tool requests', async () => {
		let capturedOptions: any[] = [];
		class FakeOpenAI {
			chat = {
				completions: {
					create: async (opts: any) => {
						capturedOptions.push(opts);
						return { choices: [{ message: { content: 'ok' } }] };
					}
				}
			};
		}
		implTestExports.setOpenAIModule?.({ default: FakeOpenAI, APIError: class extends Error { } } as any);

		const sendOnce = async (dynamicRequestConfig: any, chatMode: any) => {
			const caps = newCaptures();
			let resolveDone!: () => void;
			const done = new Promise<void>(r => { resolveDone = r; });
			await sendChatRouter({
				messages: [{ role: 'user', content: 'hi' } as any],
				separateSystemMessage: undefined,
				onText: caps.onText,
				onFinalMessage: (p) => { caps.onFinalMessage(p); resolveDone(); },
				onError: (e) => assert.fail('onError: ' + e.message),
				settingsOfProvider: { openrouter: { endpoint: 'https://openrouter.ai/api/v1', apiKey: 'sk-test' } } as any,
				modelSelectionOptions: {} as any,
				overridesOfModel: {} as any,
				modelName: 'openai/gpt-4.1',
				_setAborter: () => { },
				providerName: 'openrouter' as any,
				chatMode,
				dynamicRequestConfig,
			});
			await done;
		};

		const baseDyn = {
			endpoint: 'https://openrouter.ai/api/v1',
			apiStyle: 'openai-compatible',
			supportsSystemMessage: 'system-role',
			headers: { Authorization: 'Bearer sk-test' },
		};

		await sendOnce({ ...baseDyn, specialToolFormat: 'openai-style', parallelToolCalls: { supported: false, mode: 'safe-disabled' } }, 'agent');
		await sendOnce({ ...baseDyn, specialToolFormat: 'disabled', parallelToolCalls: { supported: true, mode: 'safe-disabled' } }, 'agent');
		await sendOnce({ ...baseDyn, specialToolFormat: 'openai-style', parallelToolCalls: { supported: true, mode: 'safe-disabled' } }, null);

		assert.strictEqual(capturedOptions.length, 3);
		for (const opts of capturedOptions) {
			assert.strictEqual(Object.prototype.hasOwnProperty.call(opts, 'parallel_tool_calls'), false);
		}
	});

	test('OpenAI-compatible requestParams override cannot inject parallel_tool_calls', async () => {
		let capturedOptions: any = null;
		class FakeOpenAI {
			chat = {
				completions: {
					create: async (opts: any) => {
						capturedOptions = opts;
						return { choices: [{ message: { content: 'ok' } }] };
					}
				}
			};
		}
		implTestExports.setOpenAIModule?.({ default: FakeOpenAI, APIError: class extends Error { } } as any);

		const caps = newCaptures();
		let resolveDone!: () => void;
		const done = new Promise<void>(r => { resolveDone = r; });
		await sendChatRouter({
			messages: [{ role: 'user', content: 'hi' } as any],
			separateSystemMessage: undefined,
			onText: caps.onText,
			onFinalMessage: (p) => { caps.onFinalMessage(p); resolveDone(); },
			onError: (e) => assert.fail('onError: ' + e.message),
			settingsOfProvider: { openrouter: { endpoint: 'https://openrouter.ai/api/v1', apiKey: 'sk-test' } } as any,
			modelSelectionOptions: {} as any,
			overridesOfModel: {} as any,
			modelName: 'openai/gpt-4.1',
			_setAborter: () => { },
			providerName: 'openrouter' as any,
			chatMode: null as any,
			dynamicRequestConfig: {
				endpoint: 'https://openrouter.ai/api/v1',
				apiStyle: 'openai-compatible',
				supportsSystemMessage: 'system-role',
				specialToolFormat: 'openai-style',
				headers: { Authorization: 'Bearer sk-test' },
				parallelToolCalls: { supported: false, mode: 'safe-disabled' },
			} as any,
			requestParams: { mode: 'override', params: { temperature: 0.5, parallel_tool_calls: true } },
		});
		await done;

		assert.strictEqual(capturedOptions.temperature, 0.5);
		assert.strictEqual(Object.prototype.hasOwnProperty.call(capturedOptions, 'parallel_tool_calls'), false);
	});

	test('OpenAI-compatible chat forwards notifyOnTruncation=false into runStream', async () => {
		setDynamicModelService({
			getDynamicCapabilities(_modelName: string) {
				return {
					contextWindow: 4096,
					reservedOutputTokenSpace: 4096,
					supportsSystemMessage: 'system-role',
					specialToolFormat: 'openai-style',
					supportsFIM: false,
					reasoningCapabilities: false,
					cost: { input: 0, output: 0 },
				};
			},
			getAllDynamicCapabilities() {
				return {};
			},
		} as any);

		let createCallCount = 0;
		class FakeOpenAI {
			chat = {
				completions: {
					create: async (_opts: any) => {
						createCallCount += 1;
						return {
							choices: [{
								finish_reason: 'length',
								message: { content: `attempt-${createCallCount}` },
							}],
						};
					},
				},
			};
		}
		implTestExports.setOpenAIModule?.({ default: FakeOpenAI, APIError: class extends Error { } } as any);

		const n = newNotificationCapture();
		const caps = newCaptures();
		let resolveDone!: () => void;
		const done = new Promise<void>(r => { resolveDone = r; });
		const onFinal: OnFinalMessage = (p) => {
			caps.onFinalMessage(p);
			resolveDone();
		};

		await sendChatRouter({
			messages: [{ role: 'user', content: 'hi' } as any],
			separateSystemMessage: undefined,
			onText: caps.onText,
			onFinalMessage: onFinal,
			onError: (e) => assert.fail('onError: ' + e.message),
			settingsOfProvider: {
				openrouter: {
					endpoint: 'https://openrouter.ai/api/v1',
					apiKey: 'sk-test',
					apiStyle: 'openai-compatible',
				}
			} as any,
			modelSelectionOptions: {} as any,
			overridesOfModel: {} as any,
			modelName: 'deepseek/deepseek-r1-0528:free',
			_setAborter: () => { },
			providerName: 'openrouter' as any,
			chatMode: null as any,
			additionalTools: undefined,
			dynamicRequestConfig: {
				endpoint: 'https://openrouter.ai/api/v1',
				apiStyle: 'openai-compatible',
				supportsSystemMessage: false as any,
				specialToolFormat: 'openai-style' as any,
				headers: { Authorization: 'Bearer sk-test', Accept: 'application/json' },
			} as any,
			requestParams: undefined,
			notificationService: n.notificationService as any,
			notifyOnTruncation: false,
		});

		await done;

		assert.strictEqual(createCallCount, 2, 'runStream should retry once on finish_reason=length');
		assert.strictEqual(n.notifications.length, 0, 'notification should respect notifyOnTruncation=false');
	});

	test('OpenAI-compatible XML mode: retries once with correction prompt when XML tool-call is malformed', async () => {
		setDynamicModelService({
			getDynamicCapabilities(_modelName: string) {
				return {
					contextWindow: 4096,
					reservedOutputTokenSpace: 4096,
					supportsSystemMessage: 'system-role',
					specialToolFormat: 'disabled',
					supportsFIM: false,
					reasoningCapabilities: false,
					cost: { input: 0, output: 0 },
				};
			},
			getAllDynamicCapabilities() {
				return {};
			},
		} as any);

		const createCalls: any[] = [];
		class FakeOpenAI {
			chat = {
				completions: {
					create: async (opts: any) => {
						createCalls.push(opts);
						const callNo = createCalls.length;

						// 1st response: malformed attribute-style XML on tool tag
						if (callNo === 1) {
							return {
								choices: [{
									message: {
										content: '<edit_file uri="./x.ts" original_snippet="old" updated_snippet="new"></edit_file>',
										tool_calls: undefined,
									},
								}],
							};
						}

						// 2nd response: valid nested-params XML
						return {
							choices: [{
								message: {
									content: [
										'<edit_file>',
										'  <uri>./x.ts</uri>',
										'  <original_snippet><![CDATA[old]]></original_snippet>',
										'  <updated_snippet><![CDATA[new]]></updated_snippet>',
										'</edit_file>',
									].join('\n'),
									tool_calls: undefined,
								},
							}],
						};
					},
				},
			};
		}
		implTestExports.setOpenAIModule?.({ default: FakeOpenAI, APIError: class extends Error { } } as any);

		const caps = newCaptures();
		let resolveDone!: () => void;
		const done = new Promise<void>(r => { resolveDone = r; });
		const onFinal: OnFinalMessage = (p) => {
			caps.onFinalMessage(p);
			resolveDone();
		};

		await sendChatRouter({
			messages: [{ role: 'user', content: 'fix ./x.ts' } as any],
			separateSystemMessage: undefined,
			onText: caps.onText,
			onFinalMessage: onFinal,
			onError: (e) => assert.fail('onError: ' + e.message),
			settingsOfProvider: {
				openrouter: {
					endpoint: 'https://openrouter.ai/api/v1',
					apiKey: 'sk-test',
					apiStyle: 'openai-compatible',
				}
			} as any,
			modelSelectionOptions: {} as any,
			overridesOfModel: {} as any,
			modelName: 'deepseek/deepseek-r1-0528:free',
			_setAborter: () => { },
			providerName: 'openrouter' as any,
			chatMode: 'agent' as any,
			additionalTools: undefined,
			dynamicRequestConfig: {
				endpoint: 'https://openrouter.ai/api/v1',
				apiStyle: 'openai-compatible',
				supportsSystemMessage: false as any,
				specialToolFormat: 'disabled' as any,
				headers: { Authorization: 'Bearer sk-test', Accept: 'application/json' },
			} as any,
			requestParams: undefined,
		});

		await done;

		assert.strictEqual(createCalls.length, 2, 'should retry once after malformed XML tool-call');
		const secondMessages = createCalls[1]?.messages ?? [];
		const correctionMsg = secondMessages[secondMessages.length - 1];
		assert.strictEqual(correctionMsg?.role, 'user', 'retry should append a user correction message');
		assert.ok(
			typeof correctionMsg?.content === 'string' && correctionMsg.content.includes('invalid XML tool call'),
			'correction prompt should explicitly mention invalid XML tool call'
		);

		const fin = caps.getFinal()!;
		assert.ok(fin?.toolCall, 'final toolCall should be parsed after retry');
		assert.strictEqual(fin.toolCall?.name, 'edit_file');
		assert.strictEqual(fin.toolCall?.isDone, true);
	});

	test('Gemini: passes thinkingConfig (budget slider) when reasoning enabled', async () => {

		implTestExports.setGetSendableReasoningInfo((_context: any, _providerName: any, _modelName: any, _modelSelectionOptions: any, _overridesOfModel: any) => {
			return { isReasoningEnabled: true, type: 'budget_slider_value', reasoningBudget: 2048 };
		});

		let capturedConfig: any = null;

		class FakeStream {
			private chunks: any[];
			constructor(chunks: any[]) { this.chunks = chunks; }
			async *[Symbol.asyncIterator]() { for (const ch of this.chunks) yield ch; }
			return(_x: any) { }
		}
		class FakeGoogleGenAI {
			constructor(_opts: any) { }
			models = {
				generateContentStream: (args: any) => {
					capturedConfig = args.config;
					const chunks = [
						{ text: 'Hello ' },
						{ text: 'world' },
						{ functionCalls: [{ name: 'read_file', args: { uri: '/b', start_line: 1, end_line: 3 }, id: 'g1' }] },
					];
					return Promise.resolve(new FakeStream(chunks));
				}
			}
		}
		implTestExports.setGoogleGenAIModule?.({ GoogleGenAI: FakeGoogleGenAI } as any);

		const caps = newCaptures();
		let resolveDone!: () => void;
		const done = new Promise<void>(r => { resolveDone = r; });
		const onFinal: OnFinalMessage = (p) => { caps.onFinalMessage(p); resolveDone(); };

		await sendChatRouter({
			messages: [{ role: 'user', parts: [{ text: 'hi' }] } as any],
			separateSystemMessage: undefined,
			onText: caps.onText,
			onFinalMessage: onFinal,
			onError: (e) => assert.fail('onError: ' + e.message),
			settingsOfProvider: { gemini: { apiKey: 'k' } } as any,
			overridesOfModel: {} as any,
			modelName: 'gemini-1.5',
			_setAborter: () => { },
			providerName: 'gemini' as any,
			modelSelectionOptions: {} as any,
			chatMode: null as any,
			additionalTools: undefined,
		});

		await done;


		assert.ok(capturedConfig && capturedConfig.thinkingConfig, 'thinkingConfig should be passed');
		assert.strictEqual(capturedConfig.thinkingConfig.thinkingBudget, 2048);

		const fin = caps.getFinal()!;
		assert.strictEqual(fin.fullText, 'Hello world');
		assert.ok(fin.toolCall);
		assert.strictEqual(fin.toolCall.name, 'read_file');
		assert.strictEqual(fin.toolCall.isDone, true);
	});
});
