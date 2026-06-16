/*--------------------------------------------------------------------------------------
 *  Copyright 2025 Glass Devtools, Inc. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt for more information.
 *--------------------------------------------------------------------------------------*/

// eslint-disable-next-line local/code-import-patterns
import * as assert from 'assert';
// eslint-disable-next-line local/code-import-patterns
import { ensureNoDisposablesAreLeakedInTestSuite } from '../../../../base/test/common/utils.js';
import { classifyToolCall, runToolCallBatch } from '../toolExecutionPolicy.js';

suite('ToolExecutionPolicy', () => {
	ensureNoDisposablesAreLeakedInTestSuite();

	test('classifies built-in tools conservatively', () => {
		assert.strictEqual(classifyToolCall({ id: '1', name: 'read_file' }).kind, 'read-only');
		assert.strictEqual(classifyToolCall({ id: '2', name: 'edit_file', rawParams: { uri: './a.ts' } }).kind, 'mutating');
		assert.strictEqual(classifyToolCall({ id: '3', name: 'run_command' }).kind, 'terminal');
		assert.strictEqual(classifyToolCall({ id: '4', name: 'acp_plan' }).kind, 'plan');
		assert.strictEqual(classifyToolCall({ id: '5', name: 'mcp__unknown' }).kind, 'unsafe-dynamic');
	});

	test('runs independent read-only calls concurrently within limit', async () => {
		let active = 0;
		let maxActive = 0;
		const order: string[] = [];

		const results = await runToolCallBatch([
			{ id: 'r1', name: 'read_file' },
			{ id: 'r2', name: 'search_for_files' },
			{ id: 'r3', name: 'ls_dir' },
		], {
			readConcurrency: 2,
			execute: async (call) => {
				active += 1;
				maxActive = Math.max(maxActive, active);
				await new Promise(r => setTimeout(r, call.id === 'r1' ? 20 : 5));
				order.push(call.id);
				active -= 1;
				return `ok:${call.id}`;
			},
			makeErrorResult: (_call, message) => `err:${message}`,
		});

		assert.strictEqual(maxActive, 2);
		assert.deepStrictEqual(results, ['ok:r1', 'ok:r2', 'ok:r3']);
		assert.deepStrictEqual(order.sort(), ['r1', 'r2', 'r3']);
	});

	test('serializes terminal and unknown dynamic tools', async () => {
		const order: string[] = [];
		await runToolCallBatch([
			{ id: 't1', name: 'run_command' },
			{ id: 'd1', name: 'mcp__write' },
		], {
			execute: async (call) => {
				order.push(`start:${call.id}`);
				await new Promise(r => setTimeout(r, 1));
				order.push(`end:${call.id}`);
				return call.id;
			},
			makeErrorResult: (_call, message) => `err:${message}`,
		});

		assert.deepStrictEqual(order, ['start:t1', 'end:t1', 'start:d1', 'end:d1']);
	});

	test('rejects duplicate same-file writes and preserves one result per call', async () => {
		const results = await runToolCallBatch([
			{ id: 'w1', name: 'edit_file', rawParams: { uri: './a.ts' } },
			{ id: 'w2', name: 'rewrite_file', rawParams: { uri: './a.ts' } },
			{ id: 'w3', name: 'edit_file', rawParams: { uri: './b.ts' } },
		], {
			execute: async (call) => `ok:${call.id}`,
			makeErrorResult: (call, message) => `err:${call.id}:${message}`,
		});

		assert.strictEqual(results[0], 'ok:w1');
		assert.ok(String(results[1]).startsWith('err:w2:'), 'duplicate write should return an error result');
		assert.strictEqual(results[2], 'ok:w3');
	});

	test('rejects a write when fingerprint changes before execution', async () => {
		let reads = 0;
		const results = await runToolCallBatch([
			{ id: 'w1', name: 'edit_file', rawParams: { uri: './a.ts' } },
		], {
			getWriteFingerprint: async () => {
				reads += 1;
				return reads === 1 ? 'before' : 'after';
			},
			execute: async (call) => `ok:${call.id}`,
			makeErrorResult: (call, message) => `err:${call.id}:${message}`,
		});

		assert.ok(String(results[0]).includes('changed after the assistant planned the edit'));
	});
});
