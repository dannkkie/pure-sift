/* tslint:disable */
/* eslint-disable */

/**
 * Chroma subsampling format
 */
export enum ChromaSampling {
    /**
     * Both vertically and horizontally subsampled.
     */
    Cs420 = 0,
    /**
     * Horizontally subsampled.
     */
    Cs422 = 1,
    /**
     * Not subsampled.
     */
    Cs444 = 2,
    /**
     * Monochrome.
     */
    Cs400 = 3,
}

export class FeaturesResult {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    readonly height: number;
    readonly keypoint_count: number;
    readonly png: Uint8Array;
    readonly width: number;
}

export class MatchAndStitchResult {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    readonly hom_inliers: number;
    readonly inliers: number;
    readonly matches_png: Uint8Array;
    readonly query_features: number;
    readonly raw_matches: number;
    readonly sim_inliers: number;
    readonly stitch_error: string | undefined;
    readonly stitch_png: Uint8Array | undefined;
    readonly target_features: number;
}

export function extract_features_png(image_bytes: Uint8Array | number[], octaves: number, scales: number, threshold: number): FeaturesResult;

export function match_and_stitch(query_bytes: Uint8Array | number[], target_bytes: Uint8Array | number[], octaves: number, scales: number, threshold: number, ratio_threshold: number, ransac_threshold: number): MatchAndStitchResult;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_featuresresult_free: (a: number, b: number) => void;
    readonly __wbg_matchandstitchresult_free: (a: number, b: number) => void;
    readonly extract_features_png: (a: number, b: number, c: number, d: number, e: number) => [number, number, number];
    readonly featuresresult_height: (a: number) => number;
    readonly featuresresult_keypoint_count: (a: number) => number;
    readonly featuresresult_png: (a: number) => [number, number];
    readonly featuresresult_width: (a: number) => number;
    readonly match_and_stitch: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => [number, number, number];
    readonly matchandstitchresult_hom_inliers: (a: number) => number;
    readonly matchandstitchresult_inliers: (a: number) => number;
    readonly matchandstitchresult_matches_png: (a: number) => [number, number];
    readonly matchandstitchresult_query_features: (a: number) => number;
    readonly matchandstitchresult_raw_matches: (a: number) => number;
    readonly matchandstitchresult_sim_inliers: (a: number) => number;
    readonly matchandstitchresult_stitch_error: (a: number) => [number, number];
    readonly matchandstitchresult_stitch_png: (a: number) => [number, number];
    readonly matchandstitchresult_target_features: (a: number) => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
