/*--------------------------------------------------------------------------------------
 *  Copyright 2025 Glass Devtools, Inc. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt for more information.
 *--------------------------------------------------------------------------------------*/

export type ToolExecutionKind =
	| 'read-only'
	| 'mutating'
	| 'terminal'
	| 'plan'
	| 'unsafe-dynamic';

export type ToolExecutionCall = {
	id: string;
	name: string;
	rawParams?: Record<string, unknown>;
	args?: Record<string, unknown>;
};

export type ToolExecutionPlan = {
	call: ToolExecutionCall;
	kind: ToolExecutionKind;
	writeTarget?: string;
};

const READ_ONLY_TOOLS = new Set([
	'read_file',
	'ls_dir',
	'get_dir_tree',
	'search_pathnames_only',
	'search_for_files',
	'search_in_file',
	'read_lint_errors',
	'activate_skill',
]);

const MUTATING_TOOLS = new Set([
	'edit_file',
	'rewrite_file',
	'create_file_or_folder',
	'delete_file_or_folder',
]);

const TERMINAL_TOOLS = new Set([
	'run_command',
]);

const PLAN_TOOLS = new Set([
	'acp_plan',
]);

const coerceRecord = (call: ToolExecutionCall): Record<string, unknown> => {
	const raw = call.rawParams ?? call.args;
	return raw && typeof raw === 'object' && !Array.isArray(raw) ? raw : {};
};

const stringifyTarget = (value: unknown): string | undefined => {
	if (typeof value === 'string' && value.trim()) return value.trim();
	if (value && typeof value === 'object') {
		const obj = value as Record<string, unknown>;
		const fsPath = obj.fsPath;
		if (typeof fsPath === 'string' && fsPath.trim()) return fsPath.trim();
		const path = obj.path;
		if (typeof path === 'string' && path.trim()) return path.trim();
	}
	return undefined;
};

export function extractToolWriteTarget(call: ToolExecutionCall): string | undefined {
	const params = coerceRecord(call);
	return stringifyTarget(params.uri)
		?? stringifyTarget(params.path)
		?? stringifyTarget(params.file_path)
		?? stringifyTarget(params.filePath)
		?? stringifyTarget(params.target)
		?? stringifyTarget(params.targetPath);
}

export function classifyToolCall(call: ToolExecutionCall): ToolExecutionPlan {
	const name = String(call.name ?? '').trim();
	if (PLAN_TOOLS.has(name)) return { call, kind: 'plan' };
	if (TERMINAL_TOOLS.has(name)) return { call, kind: 'terminal' };
	if (MUTATING_TOOLS.has(name)) return { call, kind: 'mutating', writeTarget: extractToolWriteTarget(call) };
	if (READ_ONLY_TOOLS.has(name)) return { call, kind: 'read-only' };
	return { call, kind: 'unsafe-dynamic' };
}

export function duplicateWriteTargetError(target: string | undefined): string {
	if (!target) {
		return 'Tool call was not executed because its write target could not be determined safely. Re-read the file and issue one explicit edit.';
	}
	return `Tool call was not executed because another tool call in the same assistant turn already targets "${target}". Re-read the file and issue a fresh edit.`;
}

export type ToolBatchSchedulerOptions<T> = {
	readConcurrency?: number;
	execute: (call: ToolExecutionCall, plan: ToolExecutionPlan) => Promise<T>;
	makeErrorResult: (call: ToolExecutionCall, message: string) => T | Promise<T>;
	getWriteFingerprint?: (target: string) => Promise<string | null | undefined>;
};

export async function runToolCallBatch<T>(
	calls: readonly ToolExecutionCall[],
	options: ToolBatchSchedulerOptions<T>
): Promise<T[]> {
	const plans = calls.map(classifyToolCall);
	const results: T[] = new Array(plans.length);
	const readConcurrency = Math.max(1, Math.floor(options.readConcurrency ?? 4));

	const plannedWriteFingerprints = new Map<string, string | null | undefined>();
	const seenWriteTargets = new Set<string>();
	const rejected = new Set<number>();

	for (let i = 0; i < plans.length; i += 1) {
		const plan = plans[i];
		if (plan.kind !== 'mutating') continue;
		const target = plan.writeTarget;
		if (!target) {
			results[i] = await options.makeErrorResult(plan.call, duplicateWriteTargetError(target));
			rejected.add(i);
			continue;
		}
		if (seenWriteTargets.has(target)) {
			results[i] = await options.makeErrorResult(plan.call, duplicateWriteTargetError(target));
			rejected.add(i);
			continue;
		}
		seenWriteTargets.add(target);
		if (options.getWriteFingerprint) {
			plannedWriteFingerprints.set(target, await options.getWriteFingerprint(target));
		}
	}

	const executeOne = async (index: number) => {
		if (rejected.has(index)) return;
		const plan = plans[index];
		if (plan.kind === 'mutating' && plan.writeTarget && options.getWriteFingerprint) {
			const planned = plannedWriteFingerprints.get(plan.writeTarget);
			const current = await options.getWriteFingerprint(plan.writeTarget);
			if (planned !== current) {
				results[index] = await options.makeErrorResult(
					plan.call,
					`Tool call was not executed because "${plan.writeTarget}" changed after the assistant planned the edit. Re-read the file and issue a fresh edit.`
				);
				return;
			}
		}
		results[index] = await options.execute(plan.call, plan);
	};

	let i = 0;
	while (i < plans.length) {
		if (rejected.has(i)) {
			i += 1;
			continue;
		}

		if (plans[i].kind === 'read-only') {
			const start = i;
			while (i < plans.length && plans[i].kind === 'read-only' && !rejected.has(i)) i += 1;
			const groupIndexes = Array.from({ length: i - start }, (_, offset) => start + offset);
			for (let p = 0; p < groupIndexes.length; p += readConcurrency) {
				await Promise.all(groupIndexes.slice(p, p + readConcurrency).map(executeOne));
			}
			continue;
		}

		await executeOne(i);
		i += 1;
	}

	return results;
}
