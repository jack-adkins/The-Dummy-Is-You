const DEFAULT_FOCUS_LENGTH = 1.0;
const NEAR_PLANE = 0.01;
const FAR_PLANE = 20.0;
const VIEW_ANGLE = 60.0;

export class Camera {
    constructor() {
        // Parameters
        this.viewAngle = VIEW_ANGLE;
        this.filmPlanDepth = -1.0 / Math.tan(this._deg2rad(this.viewAngle) / 2);
        this.nearPlane = NEAR_PLANE;
        this.farPlane = FAR_PLANE;
        this.screenWidth = 1;
        this.screenHeight = 1;
        this.screenWidthRatio = 1.0;
        this.rotU = this.rotV = this.rotW = 0;

        // Internal state
        this.nVec3 = vec3.create();
        this.uVec3 = vec3.create();
        this.vVec3 = vec3.create();
        this.worldToCameraMat4 = mat4.create();
        this.cameraToWorldMat4 = mat4.create();

        this.reset();
    }

    // ——— Basic Utilities ——————————————————————————————————————————

    _deg2rad(d) { return d * Math.PI / 180; }
    isNullVector(v) { return vec3.length(v) < Number.EPSILON; }

    // ——— Initialization & Parameter Setting ——————————————————————————————————

    reset() {
        this.orientLookAt(
            vec3.fromValues(0, 0, DEFAULT_FOCUS_LENGTH),
            vec3.fromValues(0, 0, 0),
            vec3.fromValues(0, 1, 0)
        );
        this.setViewAngle(VIEW_ANGLE);
        this.setNearPlane(NEAR_PLANE);
        this.setFarPlane(FAR_PLANE);
        this.screenWidthRatio = 1.0;
        this.rotU = this.rotV = this.rotW = 0;
    }

    setViewAngle(v) {
        this.viewAngle = v;
        this.filmPlanDepth = -1.0 / Math.tan(this._deg2rad(v) / 2);
    }
    setNearPlane(n) { this.nearPlane = n; }
    setFarPlane(f) { this.farPlane = f; }
    setScreenSize(w, h) {
        this.screenWidth = w;
        this.screenHeight = h;
        this.screenWidthRatio = w / h;
    }

    setRotUVW(u, v, w) {
        const du = u - this.rotU;
        const dv = v - this.rotV;
        const dw = w - this.rotW;
        this.rotateU(du);
        this.rotateV(dv);
        this.rotateW(dw);
        this.rotU = u; this.rotV = v; this.rotW = w;
    }

    // ——— Constructing Model-View Matrix ——————————————————————————————————

    orientLookAt(eyePoint, focusPoint, upVec) {
        const lookVec = vec3.sub(vec3.create(), focusPoint, eyePoint);
        this.orientLookVec(eyePoint, lookVec, upVec);
    }

    orientLookVec(eyePoint, lookVec, upVec) {
        if (this.isNullVector(lookVec) || this.isNullVector(upVec)) return;

        // 1. Calculate the three orthogonal vectors n, u, v
        const L = vec3.normalize(vec3.create(), lookVec);
        vec3.scale(this.nVec3, L, -1);
        vec3.cross(this.uVec3, L, upVec); vec3.normalize(this.uVec3, this.uVec3);
        vec3.cross(this.vVec3, this.nVec3, this.uVec3);

        // 2. Construct the orientation matrix (column-major)
        const orient = mat4.fromValues(
            this.uVec3[0], this.vVec3[0], this.nVec3[0], 0,
            this.uVec3[1], this.vVec3[1], this.nVec3[1], 0,
            this.uVec3[2], this.vVec3[2], this.nVec3[2], 0,
            0, 0, 0, 1
        );

        // 3. worldToCamera = orient * translate(-eye)
        const T1 = mat4.fromTranslation(mat4.create(), vec3.negate(vec3.create(), eyePoint));
        mat4.multiply(this.worldToCameraMat4, orient, T1);

        // 4. cameraToWorld = translate(eye) * transpose(orient)
        const orientT = mat4.transpose(mat4.create(), orient);
        const T2 = mat4.fromTranslation(mat4.create(), eyePoint);
        mat4.multiply(this.cameraToWorldMat4, T2, orientT);
    }

    getModelViewMatrix() {
        return mat4.clone(this.worldToCameraMat4);
    }
    getInverseModelViewMatrix() {
        return mat4.clone(this.cameraToWorldMat4);
    }

    // ——— Projection Related Matrices ——————————————————————————————————————————

    getScaleMatrix() {
        const s = 1.0 / Math.tan(this._deg2rad(this.viewAngle) / 2);
        const M = mat4.create();
        M[0] = s / (this.farPlane * this.screenWidthRatio);
        M[5] = s / this.farPlane;
        M[10] = 1 / this.farPlane;
        return M;
    }

    getInverseScaleMatrix() {
        const M = this.getScaleMatrix();
        M[0] = 1 / M[0];
        M[5] = 1 / M[5];
        M[10] = 1 / M[10];
        return M;
    }

    getUnhingeMatrix() {
        const c = -this.nearPlane / this.farPlane;
        const U = mat4.create(); // start as identity
        U[10] = -1 / (c + 1);
        U[14] = c / (c + 1);
        U[11] = -1;
        U[15] = 0;
        return U;
    }

    getProjectionMatrix() {
        // P = Unhinge * Scale
        return mat4.multiply(mat4.create(), this.getUnhingeMatrix(), this.getScaleMatrix());
    }

    // ——— Translation / Rotation ——————————————————————————————————————————

    translate(v) {
        // cameraToWorld = cameraToWorld * T(v)
        const T = mat4.fromTranslation(mat4.create(), v);
        mat4.multiply(this.cameraToWorldMat4, this.cameraToWorldMat4, T);
        // worldToCamera = T(-v) * worldToCamera
        const Tm = mat4.fromTranslation(mat4.create(), vec3.negate(vec3.create(), v));
        mat4.multiply(this.worldToCameraMat4, Tm, this.worldToCameraMat4);
    }

    rotate(point, axis, deg) {
        const rad = this._deg2rad(deg);

        const Tm = mat4.fromTranslation(mat4.create(), vec3.negate(vec3.create(), point));
        const R = mat4.fromRotation(mat4.create(), rad, axis);
        const Tp = mat4.fromTranslation(mat4.create(), point);

        let M = mat4.create();
        mat4.multiply(M, R, Tm);
        mat4.multiply(M, Tp, M);
        mat4.multiply(this.cameraToWorldMat4, M, this.cameraToWorldMat4);

        // 5) Synchronously update worldToCameraMat4 as the inverse matrix
        mat4.invert(this.worldToCameraMat4, this.cameraToWorldMat4);
    }

    rotateU(angle) { this.rotate(this.getEyePoint(), vec3.cross(vec3.create(), this.getLookVector(), this.getUpVector()), angle); }
    rotateV(angle) { this.rotate(this.getEyePoint(), this.getUpVector(), angle); }
    rotateW(angle) { this.rotate(this.getEyePoint(), this.getLookVector(), angle); }

    // ——— Get Camera Key Vectors ——————————————————————————————————————————

    getEyePoint() {
        // gl-matrix stores column-major, translation at indices 12,13,14
        return vec3.fromValues(
            this.cameraToWorldMat4[12],
            this.cameraToWorldMat4[13],
            this.cameraToWorldMat4[14]
        );
    }

    getLookVector() {
        // -Z column: indices 8,9,10
        return vec3.negate(vec3.create(), vec3.fromValues(
            this.cameraToWorldMat4[8],
            this.cameraToWorldMat4[9],
            this.cameraToWorldMat4[10]
        ));
    }

    getUpVector() {
        // Y column: indices 4,5,6
        return vec3.fromValues(
            this.cameraToWorldMat4[4],
            this.cameraToWorldMat4[5],
            this.cameraToWorldMat4[6]
        );
    }

    // ——— Read Parameters ——————————————————————————————————————————

    getViewAngle() { return this.viewAngle; }
    getNearPlane() { return this.nearPlane; }
    getFarPlane() { return this.farPlane; }
    getScreenWidth() { return this.screenWidth; }
    getScreenHeight() { return this.screenHeight; }
    getFilmPlanDepth() { return this.filmPlanDepth; }
    getScreenWidthRatio() { return this.screenWidthRatio; }
}