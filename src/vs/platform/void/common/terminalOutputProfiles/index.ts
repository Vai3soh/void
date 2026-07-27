/*--------------------------------------------------------------------------------------
 *  Copyright 2025 Glass Devtools, Inc. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt for more information.
 *--------------------------------------------------------------------------------------*/

import { terminalOutputLines } from '../terminalOutputSummaryModel.js';
import type { SourceRange, TerminalOutputSummary } from '../terminalOutputSummaryTypes.js';
import {
	buildOutputAdapters,
	genericBuildAdapter,
} from './buildAdapters.js';
import {
	genericPackageManagerAdapter,
	packageManagerOutputAdapters,
} from './packageManagerAdapters.js';
import {
	genericSearchListingAdapter,
	searchListingOutputAdapters,
} from './searchListingAdapters.js';
import { selectTerminalOutputAdapter } from './terminalOutputAdapterRegistry.js';
import type { TerminalOutputAdapterInput } from './terminalOutputProfileAdapter.js';
import { genericTestAdapter, testOutputAdapters } from './testAdapters.js';
import {
	genericVersionControlAdapter,
	versionControlOutputAdapters,
} from './versionControlAdapters.js';

export interface TerminalOutputAdapterOptions {
	command: string;
	rawOutput: string;
	processStatus?: 'success' | 'failure' | 'unknown';
	processStatusText?: string;
	processStatusRange?: SourceRange;
}

export type TestOutputAdapterOptions = TerminalOutputAdapterOptions;

function inputOf(options: TerminalOutputAdapterOptions): TerminalOutputAdapterInput {
	return {
		command: options.command,
		rawOutput: options.rawOutput,
		lines: terminalOutputLines(options.rawOutput),
		processStatus: options.processStatus ?? 'unknown',
		processStatusText: options.processStatusText,
		processStatusRange: options.processStatusRange,
	};
}

export function summarizeTestOutput(options: TestOutputAdapterOptions): TerminalOutputSummary {
	const input = inputOf(options);
	const selection = selectTerminalOutputAdapter(input, testOutputAdapters, genericTestAdapter);
	return selection.adapter.extract(input, selection.match);
}

export function summarizeBuildOutput(options: TerminalOutputAdapterOptions): TerminalOutputSummary {
	const input = inputOf(options);
	const selection = selectTerminalOutputAdapter(input, buildOutputAdapters, genericBuildAdapter);
	return selection.adapter.extract(input, selection.match);
}

export function summarizePackageManagerOutput(options: TerminalOutputAdapterOptions): TerminalOutputSummary {
	const input = inputOf(options);
	const selection = selectTerminalOutputAdapter(input, packageManagerOutputAdapters, genericPackageManagerAdapter);
	return selection.adapter.extract(input, selection.match);
}

export function summarizeSearchListingOutput(options: TerminalOutputAdapterOptions): TerminalOutputSummary {
	const input = inputOf(options);
	const selection = selectTerminalOutputAdapter(input, searchListingOutputAdapters, genericSearchListingAdapter);
	return selection.adapter.extract(input, selection.match);
}

export function summarizeVersionControlOutput(options: TerminalOutputAdapterOptions): TerminalOutputSummary {
	const input = inputOf(options);
	const selection = selectTerminalOutputAdapter(input, versionControlOutputAdapters, genericVersionControlAdapter);
	return selection.adapter.extract(input, selection.match);
}

export { selectTerminalOutputAdapter } from './terminalOutputAdapterRegistry.js';
export type {
	TerminalOutputAdapterInput,
	TerminalOutputAdapterMatch,
	TerminalOutputAdapterSelection,
	TerminalOutputProfileAdapter,
} from './terminalOutputProfileAdapter.js';
export {
	buildOutputAdapters,
	compilerRustcBuildAdapter,
	genericBuildAdapter,
	typescriptEslintBuildAdapter,
} from './buildAdapters.js';
export {
	genericPackageManagerAdapter,
	npmPnpmYarnPackageAdapter,
	packageManagerOutputAdapters,
} from './packageManagerAdapters.js';
export {
	catInspectionAdapter,
	findLsTreeListingAdapter,
	genericSearchListingAdapter,
	rgGrepSearchAdapter,
	searchListingOutputAdapters,
} from './searchListingAdapters.js';
export {
	cargoTestAdapter,
	genericTestAdapter,
	goTestAdapter,
	jestLikeTestAdapter,
	pytestTestAdapter,
	testOutputAdapters,
} from './testAdapters.js';
export {
	genericVersionControlAdapter,
	gitDiffAdapter,
	gitLogAdapter,
	gitMutationAdapter,
	gitStatusAdapter,
	versionControlOutputAdapters,
} from './versionControlAdapters.js';
