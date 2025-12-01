import { TransformationType } from './SceneDataStructures.js';

/**
 * SceneFlattener
 *   - constructor(rootNode: SceneNode)
 *   - flatten(): flatten tree and pack into Float32Array
 *   - getFloat32Array(): returns the packed Float32Array
 *   - getObjectCount(): returns number of objects flattened
 */
export class SceneFlattener {
    constructor(rootNode) {
        this.rootNode = rootNode;
        this.objectList = [];
        this.floatsPerObject = 1 + 16 + 18; // type + 4x4 matrix + material(18)
        this._dataArray = null;
        this._flattened = false;
    }

    flatten() {
        if (!this.rootNode) {
            console.error("SceneFlattener: rootNode is null.");
            return;
        }
        this.objectList = [];
        const identityMat = new Float32Array([
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        ]);
        this._traverseNode(this.rootNode, identityMat);
        this._indexTextureMaps(); // new for a4 
        this._buildDataArray();
        this._flattened = true;
    }

    // new for a4 - handle texture mapping 
    // build a list of unique textures and assign them indices 
    _indexTextureMaps() {
        this.textureMaps = [];
        for (let obj of this.objectList) {
            const map = obj.material.textureMap;
            if (!map.isUsed) {
                continue;
            }
            let idx = this.textureMaps.findIndex(m => m.filename === map.filename);
            if (idx < 0) {
                idx = this.textureMaps.length;
                this.textureMaps.push(map);
            }
            map.textureIndex = idx;
        }
    }

    getTextureMaps() {
        return this.textureMaps || []; 
    }

    getObjectCount() {
        return this.objectList.length;
    }

    getFloat32Array() {
        if (!this._flattened) {
            console.warn("SceneFlattener: call flatten() first.");
        }
        return this._dataArray;
    }

    _traverseNode(node, parentMat) {
        // 1) copy parentMat
        const currentMat = parentMat.slice();

        // 2) apply node.transformations sequentially
        for (const t of node.transformations) {
            const T = this._buildTransformMatrix(t);
            this._multiplyMat4(currentMat, T, currentMat);
        }

        // 3) if has primitives, add each as a flat object
        if (node.primitives && node.primitives.length > 0) {
            for (const prim of node.primitives) {
                this.objectList.push({
                    type: prim.type,
                    worldMatrix: currentMat.slice(),
                    material: prim.material
                });
            }
        }

        // 4) recurse children
        for (const child of node.children) {
            this._traverseNode(child, currentMat);
        }
    }

    _buildTransformMatrix(t) {
        const M = new Float32Array(16);
        M.set([
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        ]);

        switch (t.type) {
            case TransformationType.TRANSFORMATION_TRANSLATE: {
                const [x, y, z] = t.translate;
                // row-major translation
                M[3] = x; M[7] = y; M[11] = z;
                break;
            }
            case TransformationType.TRANSFORMATION_SCALE: {
                const [sx, sy, sz] = t.scale;
                M[0] = sx; M[5] = sy; M[10] = sz;
                break;
            }
            case TransformationType.TRANSFORMATION_ROTATE: {
                const [x, y, z] = t.rotate;
                const angle = t.angle;
                const len = Math.hypot(x, y, z);
                if (len < 1e-6) break;
                const ux = x / len, uy = y / len, uz = z / len;
                const c = Math.cos(angle), s = Math.sin(angle);
                // row-major Rodrigues
                M[0] = c + ux * ux * (1 - c);
                M[1] = ux * uy * (1 - c) - uz * s;
                M[2] = ux * uz * (1 - c) + uy * s;
                M[3] = 0;
                M[4] = uy * ux * (1 - c) + uz * s;
                M[5] = c + uy * uy * (1 - c);
                M[6] = uy * uz * (1 - c) - ux * s;
                M[7] = 0;
                M[8] = uz * ux * (1 - c) - uy * s;
                M[9] = uz * uy * (1 - c) + ux * s;
                M[10] = c + uz * uz * (1 - c);
                M[11] = 0;
                M[12] = 0; M[13] = 0; M[14] = 0; M[15] = 1;
                break;
            }
            case TransformationType.TRANSFORMATION_MATRIX: {
                // t.matrix is assumed row-major from parser
                M.set(t.matrix);
                break;
            }
            default:
                break;
        }

        return M;
    }

    _multiplyMat4(A, B, out) {
        // row-major multiply: out = A * B
        const a = A, b = B;
        const o = new Float32Array(16);
        for (let r = 0; r < 4; r++) {
            for (let c = 0; c < 4; c++) {
                let sum = 0;
                for (let k = 0; k < 4; k++) {
                    sum += a[r * 4 + k] * b[k * 4 + c];
                }
                o[r * 4 + c] = sum;
            }
        }
        out.set(o);
    }

    _buildDataArray() {
        const N = this.objectList.length;
        const fpo = this.floatsPerObject;
        this._dataArray = new Float32Array(N * fpo);
        let offset = 0;
        for (let i = 0; i < N; i++) {
            const obj = this.objectList[i];
            // type
            this._dataArray[offset++] = obj.type;
            // worldMatrix (16 floats)
            const wm = obj.worldMatrix;
            for (let j = 0; j < 16; j++) {
                this._dataArray[offset++] = wm[j];
            }
            // material (18 floats)
            const m = obj.material;
            this._dataArray[offset++] = m.cAmbient.r;
            this._dataArray[offset++] = m.cAmbient.g;
            this._dataArray[offset++] = m.cAmbient.b;
            this._dataArray[offset++] = m.cDiffuse.r;
            this._dataArray[offset++] = m.cDiffuse.g;
            this._dataArray[offset++] = m.cDiffuse.b;
            this._dataArray[offset++] = m.cSpecular.r;
            this._dataArray[offset++] = m.cSpecular.g;
            this._dataArray[offset++] = m.cSpecular.b;
            this._dataArray[offset++] = m.shininess;
            this._dataArray[offset++] = m.ior;
            this._dataArray[offset++] = (m.textureMap.isUsed ? 1.0 : 0.0);
            this._dataArray[offset++] = m.textureMap.repeatU;
            this._dataArray[offset++] = m.textureMap.repeatV;
            this._dataArray[offset++] = m.textureMap.textureIndex;
            this._dataArray[offset++] = m.cReflective.r;
            this._dataArray[offset++] = m.cReflective.g;
            this._dataArray[offset++] = m.cReflective.b;
        }
    }
}