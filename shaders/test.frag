#version 300 es
precision highp float;

// IMPORTANT: Most of the parameters are passed from render.js. You can always modify any of them.
// You can check how the buffers are constructed in SceneFlattener.js

// Screen resolution in pixels
uniform vec2  uResolution;

// Camera data
uniform vec3 uCameraPos;
uniform mat4 uCamWorldMatrix;

// Global material coefficients
uniform float uGlobalKa;
uniform float uGlobalKd;
uniform float uGlobalKs;
uniform float uGlobalKt;

// Scene data stored in a 2D RGBA32F texture
uniform sampler2D uSceneBuffer; // width = floatsPerRow/4, height = objectCount
uniform int       uObjectCount;     // number of objects (rows in texture)
uniform int       uFloatsPerRow; // floats per row (32-bit floats)
uniform int       uSceneTexWidth;   // texture width = ceil(floatsPerObject/4)
uniform int       uSceneTexHeight;  // texture height = objectCount
uniform sampler2D uTextures[8]; // up to 8 distince textures if needed 

// Light data arrays
// NOTE: not all fields are useful
uniform int   uNumLights;
uniform int   uLightType[16];
uniform vec3  uLightColor[16];
uniform vec3  uLightPos[16];
uniform vec3  uLightDir[16];
uniform float uLightRadius[16];
uniform float uLightPenumbra[16];
uniform float uLightAngle[16];
uniform float uLightWidth[16];
uniform float uLightHeight[16];

uniform int uMaxDepth; // maximum recursion depth for reflections 

// constants
const float EPSILON = 1e-3;
const float PI = 3.141592653589793;

// TODO: This should be your output color, instead of gl_FragColor
out vec4 outColor;

/*********** Helper Functions **********/

// ----------------------------------------------
// fetchFloat: retrieve a single float from uSceneBuffer
// idx = index of that float within the object's flattened data
// row = which object (row index) to fetch from
float fetchFloat(int idx, int row) {
    // Calculate which texel (column) and channel (RGBA) to read
    int texelX  = idx / 4;          // one texel holds 4 floats
    int channel = idx - texelX * 4; // idx % 4

    // Fetch the texel once
    vec4 texel = texelFetch(uSceneBuffer, ivec2(texelX, row), 0);

    // Return the appropriate component
    if (channel == 0) return texel.r;
    if (channel == 1) return texel.g;
    if (channel == 2) return texel.b;
    return texel.a;
}

// ----------------------------------------------
// fetchWorldMatrix: reconstruct a 4×4 world transform matrix for object idx
// Each object stores 1 type float + 16 matrix floats + 12 material floats, total = uFloatsPerRow
mat4 fetchWorldMatrix(int idx) {
    mat4 M = mat4(1.0);

    // Base index in flattened array for this object
    int base = 1;
    // +1 skips the type code; next 16 floats are the world matrix in row-major order

    // Loop over rows and columns to assemble the mat4 (column-major in GLSL)
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            float value = fetchFloat(base + r * 4 + c, idx);
            M[c][r] = value;
        }
    }
    return M;
}

// ----------------------------------------------
// Material struct to hold 12 floats of material data
struct Material {
    vec3 ambientColor;
    vec3 diffuseColor;
    vec3 specularColor;
    float shininess;
    float ior;
    float useTexture;
    vec2 repeatUV;
    float textureIndex;
    vec3 reflectiveColor;
};

// fetchMaterial: reconstruct the material attributes for object idx
Material fetchMaterial(int idx) {
    Material mat;

    // Base index for material data: skip type (1) + matrix (16)
    int base = 1 + 16;

    mat.ambientColor.r  = fetchFloat(base + 0,  idx);
    mat.ambientColor.g  = fetchFloat(base + 1,  idx);
    mat.ambientColor.b  = fetchFloat(base + 2,  idx);

    mat.diffuseColor.r  = fetchFloat(base + 3,  idx);
    mat.diffuseColor.g  = fetchFloat(base + 4,  idx);
    mat.diffuseColor.b  = fetchFloat(base + 5,  idx);

    mat.specularColor.r = fetchFloat(base + 6,  idx);
    mat.specularColor.g = fetchFloat(base + 7,  idx);
    mat.specularColor.b = fetchFloat(base + 8,  idx);

    mat.shininess       = fetchFloat(base + 9,  idx);
    mat.ior             = fetchFloat(base + 10, idx);

    mat.useTexture      = fetchFloat(base + 11, idx);
    mat.repeatUV.x      = fetchFloat(base + 12, idx);
    mat.repeatUV.y      = fetchFloat(base + 13, idx);
    mat.textureIndex    = fetchFloat(base + 14, idx);

    mat.reflectiveColor = vec3(
      fetchFloat(base + 15, idx),
      fetchFloat(base + 16, idx),
      fetchFloat(base + 17, idx)
    );

    return mat;
}

// ----------------------------------------------
// intersectSphere: ray-sphere intersection in object space
// Sphere is centered at origin with radius = 0.5
float intersectSphere(vec3 ro, vec3 rd) {
    // TODO: implement ray-sphere intersection
    return -1.0;
}

// ----------------------------------------------
// normalSphere: compute normal at intersection point in object space
vec3 normalSphere(vec3 hitPos) {
    // TODO: implement normal computation for sphere
    return vec3(1.0);
}

// ----------------------------------------------
// intersectCube: ray-cube intersection in object space
// Cube is centered at origin with side length = 1
float intersectCube(vec3 ro, vec3 rd) {
    // TODO: implement ray-cube intersection
    return -1.0;
}

// ----------------------------------------------
// normalCube: compute normal at intersection point in object space
vec3 normalCube(vec3 hitPos) {
    // TODO: implement normal computation for cube
    return vec3(1.0);
}

// ----------------------------------------------
// intersectCylinder: ray-cylinder intersection in object space
float intersectCylinder(vec3 ro, vec3 rd) {
    float t = -1.0;

    // TODO: implement ray-cylinder intersection
    // Cylinder is centered at origin, radius = 0.5, height = 1

    return t; // return closest intersection distance, or -1.0 if no hit
}

// ----------------------------------------------
// normalCylinder: compute normal at intersection point in object space
vec3 normalCylinder(vec3 hitPos) {
    // TODO: implement normal computation for cylinder
    // Cylinder is centered at origin, radius = 0.5, height = 1
    return vec3(1.0);
}

// ----------------------------------------------
// intersectCone: ray-cone intersection in object space
float intersectCone(vec3 ro, vec3 rd) {
    // TODO: implement ray-cone intersection
    return -1.0;
}

// ----------------------------------------------
// normalCone: compute normal at intersection point in object space
vec3 normalCone(vec3 hitPos) {
    // TODO: implement normal computation for cone
    return vec3(1.0);
}

vec2 getTexCoordSphere(vec3 hit, vec2 repeatUV) {
    // TODO: implement spherical mapping
    return vec2(0.0);
}

vec2 getTexCoordCube(vec3 hit, vec3 dominantFace, vec2 repeatUV) {
    // TODO: implement cubic mapping
    return vec2(0.0);
}

vec2 getTexCoordCylinder(vec3 hit, vec2 repeatUV) {
    // TODO: implement cylindrical mapping
    return vec2(0.0);
}

vec2 getTexCoordCone(vec3 hit, vec2 repeatUV) {
    // TODO: implement conical mapping
    return vec2(0.0);
}


// ----------------------------------------------
// getWorldRayDir: reconstruct world-space ray direction using uCamWorldMatrix
vec3 getWorldRayDir() {
    vec2 uv  = gl_FragCoord.xy / uResolution; 
    // TODO: compute ray direction in world space
    vec3 dir = vec3(0.0);
    return dir;
}

// to help test occlusion (shadow)
bool isInShadow(vec3 p, vec3 lightDir, float maxDist) {
    // TODO: implement shadow ray intersection test
    return false; 
}


// bounce = recursion level (0 for primary rays)
vec3 traceRay(vec3 rayOrigin, vec3 rayDir) {
    // TODO: implement ray tracing logic
    return vec3(0.0);
}


// ----------------------------------------------
// main: iterate over all objects, test intersection, and shade
void main() {
    // Compute ray origin and direction in world space
    vec3 rayOrigin = uCameraPos;
    vec3 rayDir    = getWorldRayDir();

    // process and get final color 
    vec3 color = traceRay(rayOrigin, rayDir);
    outColor = vec4(color, 1.0);
}