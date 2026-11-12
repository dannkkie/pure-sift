/* @ts-self-types="./pure_sift_wasm_demo.d.ts" */

/**
 * Chroma subsampling format
 * @enum {0 | 1 | 2 | 3}
 */
export const ChromaSampling = Object.freeze({
    /**
     * Both vertically and horizontally subsampled.
     */
    Cs420: 0, "0": "Cs420",
    /**
     * Horizontally subsampled.
     */
    Cs422: 1, "1": "Cs422",
    /**
     * Not subsampled.
     */
    Cs444: 2, "2": "Cs444",
    /**
     * Monochrome.
     */
    Cs400: 3, "3": "Cs400",
});

export class FeaturesResult {
    static __wrap(ptr) {
        const obj = Object.create(FeaturesResult.prototype);
        obj.__wbg_ptr = ptr;
        FeaturesResultFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        FeaturesResultFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_featuresresult_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get height() {
        const ret = wasm.featuresresult_height(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get keypoint_count() {
        const ret = wasm.featuresresult_keypoint_count(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {Uint8Array}
     */
    get png() {
        const ret = wasm.featuresresult_png(this.__wbg_ptr);
        var v1 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        return v1;
    }
    /**
     * @returns {number}
     */
    get width() {
        const ret = wasm.featuresresult_width(this.__wbg_ptr);
        return ret >>> 0;
    }
}
if (Symbol.dispose) FeaturesResult.prototype[Symbol.dispose] = FeaturesResult.prototype.free;

export class MatchAndStitchResult {
    static __wrap(ptr) {
        const obj = Object.create(MatchAndStitchResult.prototype);
        obj.__wbg_ptr = ptr;
        MatchAndStitchResultFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        MatchAndStitchResultFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_matchandstitchresult_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get hom_inliers() {
        const ret = wasm.matchandstitchresult_hom_inliers(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get inliers() {
        const ret = wasm.matchandstitchresult_inliers(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {Uint8Array}
     */
    get matches_png() {
        const ret = wasm.matchandstitchresult_matches_png(this.__wbg_ptr);
        var v1 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        return v1;
    }
    /**
     * @returns {number}
     */
    get query_features() {
        const ret = wasm.matchandstitchresult_query_features(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get raw_matches() {
        const ret = wasm.matchandstitchresult_raw_matches(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get sim_inliers() {
        const ret = wasm.matchandstitchresult_sim_inliers(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {string | undefined}
     */
    get stitch_error() {
        const ret = wasm.matchandstitchresult_stitch_error(this.__wbg_ptr);
        let v1;
        if (ret[0] !== 0) {
            v1 = getStringFromWasm0(ret[0], ret[1]);
            wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        }
        return v1;
    }
    /**
     * @returns {Uint8Array | undefined}
     */
    get stitch_png() {
        const ret = wasm.matchandstitchresult_stitch_png(this.__wbg_ptr);
        let v1;
        if (ret[0] !== 0) {
            v1 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
            wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        }
        return v1;
    }
    /**
     * @returns {number}
     */
    get target_features() {
        const ret = wasm.matchandstitchresult_target_features(this.__wbg_ptr);
        return ret >>> 0;
    }
}
if (Symbol.dispose) MatchAndStitchResult.prototype[Symbol.dispose] = MatchAndStitchResult.prototype.free;

/**
 * @param {Uint8Array | number[]} image_bytes
 * @param {number} octaves
 * @param {number} scales
 * @param {number} threshold
 * @returns {FeaturesResult}
 */
export function extract_features_png(image_bytes, octaves, scales, threshold) {
    const val2 = +octaves;
    const val3 = +scales;
    const val4 = +threshold;
    const ptr0 = passArray8ToWasm0(image_bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.extract_features_png(ptr0, len0, val2, val3, val4);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return FeaturesResult.__wrap(ret[0]);
}

/**
 * @param {Uint8Array | number[]} query_bytes
 * @param {Uint8Array | number[]} target_bytes
 * @param {number} octaves
 * @param {number} scales
 * @param {number} threshold
 * @param {number} ratio_threshold
 * @param {number} ransac_threshold
 * @returns {MatchAndStitchResult}
 */
export function match_and_stitch(query_bytes, target_bytes, octaves, scales, threshold, ratio_threshold, ransac_threshold) {
    const val3 = +octaves;
    const val4 = +scales;
    const val5 = +threshold;
    const val6 = +ratio_threshold;
    const val7 = +ransac_threshold;
    let ptr0, len0, ptr1, len1;
    try {
        ptr0 = passArray8ToWasm0(query_bytes, wasm.__wbindgen_malloc);
        len0 = WASM_VECTOR_LEN;
        ptr1 = passArray8ToWasm0(target_bytes, wasm.__wbindgen_malloc);
        len1 = WASM_VECTOR_LEN;
    } catch (e) {
        if (ptr1) { wasm.__wbindgen_free(ptr1, len1 * 1, 1); }
        if (ptr0) { wasm.__wbindgen_free(ptr0, len0 * 1, 1); }
        throw e;
    }
    const ret = wasm.match_and_stitch(ptr0, len0, ptr1, len1, val3, val4, val5, val6, val7);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return MatchAndStitchResult.__wrap(ret[0]);
}
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg___wbindgen_throw_4c88d8bf264fd984: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbindgen_generic_0000000000000001: function(arg0, arg1) {
            // Cast intrinsic for `Ref(String) -> Externref`.
            const ret = getStringFromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            table.set(0, undefined);
            table.set(1024, undefined);
            table.set(1025, null);
            table.set(1026, true);
            table.set(1027, false);
        },
    };
    return {
        __proto__: null,
        "./pure_sift_wasm_demo_bg.js": import0,
    };
}

const FeaturesResultFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_featuresresult_free(ptr, 1));
const MatchAndStitchResultFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_matchandstitchresult_free(ptr, 1));

function getArrayU8FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return decodeText(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function passArray8ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 1, 1) >>> 0;
    getUint8ArrayMemory0().set(arg, ptr / 1);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_externrefs.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(bytes) {
    numBytesDecoded += bytes.length;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true });
        cachedTextDecoder.decode();
        numBytesDecoded = bytes.length;
    }
    return cachedTextDecoder.decode(bytes);
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasmInstance, wasm;
function __wbg_finalize_init(instance, module) {
    wasmInstance = instance;
    wasm = instance.exports;
    wasmModule = module;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (!module.ok) {
            throw new Error(`failed to fetch Wasm: ${module.status} ${module.statusText} fetching '${module.url}'`);
        }

        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

let __wbg_init_promise;
async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;
    if (__wbg_init_promise === undefined) {
        __wbg_init_promise = __wbg_init_once(module_or_path);
        __wbg_init_promise.catch(() => { __wbg_init_promise = undefined; });
    }
    return __wbg_init_promise;
}

async function __wbg_init_once(module_or_path) {

    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('pure_sift_wasm_demo_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
