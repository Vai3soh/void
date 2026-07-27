/*--------------------------------------------------------------------------------------
 *  Copyright 2025 Glass Devtools, Inc. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt for more information.
 *--------------------------------------------------------------------------------------*/

/**
 * Shared typed contracts for the terminal output quality-upgrade pipeline.
 *
 * These types are pure and platform-agnostic: no imports from VS Code services,
 * no `any`, and no fabricated values. They are used by the classifier, policy,
 * adapters, reducer, and renderer.
 */

/**
 * Known terminal output families. The classifier selects one of these based on
 * command header + content evidence. When evidence is insufficient the result
 * is `'generic'`.
 */
export type TerminalOutputProfile =
	| 'test'
	| 'build-diagnostics'
	| 'package-manager'
	| 'search-listing'
	| 'version-control'
	| 'logs'
	| 'generic';

/**
 * Why a summary was triggered.
 * - `hard-limit`: raw output exceeds `maxToolOutputLength`; summary is mandatory.
 * - `verbose`: raw is shorter than hard-limit but a safe candidate saves significant length.
 */
export type SummaryReason = 'hard-limit' | 'verbose';

/**
 * Confidence level assigned by the classifier (or adapter) to the selected profile.
 */
export type SummaryConfidence = 'high' | 'medium' | 'low';

/**
 * 1-based range of source lines in the original (normalized) raw output.
 */
export interface SourceRange {
	/** 1-based inclusive start line. */
	startLine: number;
	/** 1-based inclusive end line. */
	endLine: number;
}

/**
 * A single typed count fact extracted from native output (e.g. "10 passed").
 * The value is `undefined` when the count is referenced but could not be parsed
 * from the raw evidence - NEVER fabricated as zero.
 */
export interface CountFact {
	/** Human-readable name of the count (e.g. "passed", "failed", "warnings"). */
	name: string;
	/** Exact integer value when parsed; `undefined` when present but unparseable. */
	value: number | undefined;
	/** Where in the raw output this fact was derived from. */
	sourceRange: SourceRange;
}

/**
 * A block of summary content: can be a verbatim snippet, a diagnostic, an
 * aggregate marker, or a sample block.
 */
export type SummaryBlockKind = 'status' | 'native-summary' | 'diagnostic' | 'warning' | 'aggregate' | 'sample' | 'overview';

export interface SummaryBlock {
	kind: SummaryBlockKind;
	/** User-visible lines in the summary for this block. */
	lines: readonly string[];
	/** 1-based source range(s) this block covers. May be empty for synthetic markers. */
	sourceRanges: readonly SourceRange[];
	/** If true, this block MUST survive budget reduction (e.g. failure identity). */
	protected: boolean;
}

/**
 * A single diagnostic (error/warning from a build, lint, or test run).
 */
export interface DiagnosticBlock {
	/** File path from raw output (may be undefined for project-level diagnostics). */
	file: string | undefined;
	/** 1-based line number in the file, if present. */
	line: number | undefined;
	/** 1-based column number in the file, if present. */
	column: number | undefined;
	/** Diagnostic code/rule identifier (e.g. "TS2345", "no-unused-vars"). */
	code: string | undefined;
	/** The primary diagnostic message. */
	message: string;
	/** Severity: "error" or "warning". */
	severity: 'error' | 'warning';
	/** Raw lines providing context (code frame, note, help). */
	contextLines: readonly string[];
	/** Source range in the raw output for this diagnostic and its context. */
	sourceRange: SourceRange;
}

/**
 * Evidence collected before reduction.
 */
export interface SummaryEvidence {
	/** Raw output lines that look like a native final summary. */
	nativeSummaryLines: readonly string[];
	/** Raw output lines indicating process exit/final status. */
	statusLines: readonly string[];
	/** Count facts extracted so far (may be extended by adapters). */
	countFacts: readonly CountFact[];
	/** Diagnostic blocks identified before reduction. */
	diagnostics: readonly DiagnosticBlock[];
	/** Ranges in the raw output that must not be touched by generic reduction. */
	protectedRanges: readonly SourceRange[];
}

/**
 * Marker evidence used to select a profile. Counts represent distinct matched
 * marker patterns, so repeated weak lines cannot inflate confidence.
 */
export interface ProfileClassificationEvidence {
	commandMarkerCount: number;
	contentMarkerCount: number;
	uniqueContentMarkerCount: number;
	contentRanges: readonly SourceRange[];
	uniqueContentRanges: readonly SourceRange[];
	ambiguousProfiles: readonly TerminalOutputProfile[];
}

/**
 * Result of profile classification.
 */
export interface ClassificationResult {
	profile: TerminalOutputProfile;
	confidence: SummaryConfidence;
	/** Human-readable label for the selected family adapter. */
	adapter: string;
	/** Marker evidence supporting the selected profile. */
	classificationEvidence: ProfileClassificationEvidence;
	/** Structured raw evidence collected before reduction. */
	evidence: SummaryEvidence;
}

/**
 * Full typed summary model produced by the pipeline.
 */
export interface TerminalOutputSummary {
	profile: TerminalOutputProfile;
	adapter: string;
	confidence: SummaryConfidence;
	status: 'success' | 'failure' | 'unknown';
	counts: readonly CountFact[];
	blocks: readonly SummaryBlock[];
	protectedRanges: readonly SourceRange[];
}

/**
 * Measured evidence that a short output is verbose rather than already compact.
 */
export interface TerminalOutputVerboseEvidence {
	rawLineCount: number;
	repeatNoiseRatio: number;
	progressDensity: number;
	blankProgressChurnRatio: number;
	hasNativeSummary: boolean;
	profileConfidence: SummaryConfidence;
}

/**
 * Fully rendered candidate evaluated by the adaptive policy. `text` includes
 * all model-facing summary overhead, so savings are measured truthfully.
 */
export interface TerminalOutputSummaryCandidate {
	summary: TerminalOutputSummary;
	text: string;
	mandatorySignals: readonly string[];
	verboseEvidence: TerminalOutputVerboseEvidence;
}

/**
 * Result of the adaptive policy decision.
 */
export type AdaptiveDecision =
	| { kind: 'pass-through' }
	| { kind: 'verbose'; candidate: TerminalOutputSummaryCandidate; savedChars: number }
	| { kind: 'hard-limit'; candidate: TerminalOutputSummaryCandidate | undefined };

/**
 * Options for the summary pipeline, extending the legacy summarizer options
 * with quality-upgrade fields.
 */
export interface SummaryPipelineOptions {
	/** Raw output as a single string. */
	rawOutput: string;
	/** Command that produced this output (e.g. "npm test", "git push"). */
	command: string;
	/** Hard character limit for the model-facing result. */
	maxToolOutputLength: number;
	/** Hint for how many head lines to sample when budget allows. */
	headLines: number;
	/** Hint for how many tail lines to sample when budget allows. */
	tailLines: number;
}

/** Verbose savings threshold constants. */
export const VERBOSE_SAVINGS_MIN_ABSOLUTE = 1000;
export const VERBOSE_SAVINGS_MIN_FRACTION = 0.25;

/** Conservative defaults for accepting a short verbose candidate. */
export const VERBOSE_REPEAT_NOISE_MIN_RATIO = 0.2;
export const VERBOSE_PROGRESS_MIN_DENSITY = 0.2;
export const VERBOSE_BLANK_PROGRESS_MIN_CHURN_RATIO = 0.25;
export const VERBOSE_NATIVE_SUMMARY_MIN_LINES = 40;
