/*--------------------------------------------------------------------------------------
 *  Copyright 2025 Glass Devtools, Inc. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt for more information.
 *--------------------------------------------------------------------------------------*/

import assert from 'assert';
import { Event } from '../../../../../base/common/event.js';
import { URI } from '../../../../../base/common/uri.js';
import { ensureNoDisposablesAreLeakedInTestSuite } from '../../../../../base/test/common/utils.js';
import { ToolsService } from '../../browser/toolsService.js';

suite('ToolsService - Agent Skills', () => {
	ensureNoDisposablesAreLeakedInTestSuite();

	function createToolsService(agentSkillsService: any) {
		const workspaceContextService: any = {
			getWorkspace() {
				return { folders: [{ uri: URI.file('/workspace') }] };
			},
		};
		const instantiationService: any = {
			createInstance() {
				return {
					file: () => { throw new Error('not used'); },
					text: () => { throw new Error('not used'); },
				};
			},
		};

		return new ToolsService(
			{},
			workspaceContextService,
			{},
			instantiationService,
			{},
			{},
			{},
			{ getStreamState: () => 'idle' },
			{},
			{ read: () => [] },
			{ state: { globalSettings: { includeToolLintErrors: false } } },
			agentSkillsService,
		);
	}

	test('activate_skill validates name and returns structured content', async () => {
		const activation = {
			name: 'code-review',
			body: 'Review carefully.',
			skillFileUri: URI.file('/workspace/.void/skills/code-review/SKILL.md'),
			skillDirUri: URI.file('/workspace/.void/skills/code-review'),
			resources: [],
			diagnostics: [],
			contentForModel: '<skill_content name="code-review">Review carefully.</skill_content>',
		};
		const svc = createToolsService({
			onDidChangeCatalog: Event.None,
			activateSkill: async (name: string) => {
				assert.strictEqual(name, 'code-review');
				return activation;
			},
		});

		const params = svc.validateParams.activate_skill({ name: 'code-review' } as any);
		const { result } = await svc.callTool.activate_skill(params);
		const resolved = await result;

		assert.strictEqual(resolved.name, 'code-review');
		assert.strictEqual(svc.stringOfResult.activate_skill(params, resolved), activation.contentForModel);
	});

	test('activate_skill rejects unknown skills through service error', async () => {
		const svc = createToolsService({
			onDidChangeCatalog: Event.None,
			activateSkill: async () => {
				throw new Error('Skill "missing" is unavailable.');
			},
		});

		const params = svc.validateParams.activate_skill({ name: 'missing' } as any);
		await assert.rejects(() => svc.callTool.activate_skill(params), /unavailable/);
	});
});
