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

const int MAX_LIGHTS = 16;  // soft upper limit for number of lightPos
const int MAX_OBJECTS = 256; // soft upper limit for number of objects

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
    // quadratic coefficients for (ro + t * rd)^2 = R^2, where R = 0.5
    float radius = 0.5;
    float A = dot(rd, rd);
    float B = 2.0 * dot(ro, rd);
    float C = dot(ro, ro) - radius * radius;

    float disc = B * B - 4.0 * A * C;
    if (disc < 0.0) return -1.0;

    float sqrtDisc = sqrt(disc);
    float t1 = (-B + sqrtDisc) / (2.0 * A);
    float t2 = (-B - sqrtDisc) / (2.0 * A);

    float tMin = 1e20;

    if (t1 > EPSILON) tMin = min(tMin, t1);
    if (t2 > EPSILON) tMin = min(tMin, t2);

    if (tMin < 1e19) {
        return tMin;
    }
    return -1.0;
}

// ----------------------------------------------
// normalSphere: compute normal at intersection point in object space
vec3 normalSphere(vec3 hitPos) {
    // For a sphere centered at the origin, normal is just the normalized position
    return normalize(hitPos);
}

// ----------------------------------------------
// intersectCube: ray-cube intersection in object space
// Cube is centered at origin with side length = 1
float intersectCube(vec3 ro, vec3 rd) {
    // t for planes on each axis
    vec3 tMin = (vec3(-0.5) - ro) / rd;
    vec3 tMax = (vec3( 0.5) - ro) / rd;

    // For each axis, t1 is near, t2 is far
    vec3 t1 = min(tMin, tMax);
    vec3 t2 = max(tMin, tMax);

    // Entry and exit distances
    float tNear = max(max(t1.x, t1.y), t1.z);
    float tFar  = min(min(t2.x, t2.y), t2.z);

    // No hit if we exit before we enter, or everything is behind us
    if (tFar < EPSILON || tNear > tFar) {
        return -1.0;
    }

    // If we start inside the box (tNear < 0), use tFar to exit
    float tHit = (tNear > EPSILON) ? tNear : tFar;

    return (tHit > EPSILON) ? tHit : -1.0;
}

// ----------------------------------------------
// normalCube: compute normal at intersection point in object space
vec3 normalCube(vec3 p) {
    float ax = abs(p.x);
    float ay = abs(p.y);
    float az = abs(p.z);

    // snap to whichever component has largest magnitude
    if (ax >= ay && ax >= az) {
        return vec3(p.x > 0.0 ? 1.0 : -1.0, 0.0, 0.0);
    } else if (ay >= ax && ay >= az) {
        return vec3(0.0, p.y > 0.0 ? 1.0 : -1.0, 0.0);
    } else {
        return vec3(0.0, 0.0, p.z > 0.0 ? 1.0 : -1.0);
    }
}

// ----------------------------------------------
// intersectCylinder: ray-cylinder intersection in object space
float intersectCylinder(vec3 ro, vec3 rd) {
    float radius = 0.5;

    float a = rd.x * rd.x + rd.z * rd.z;
    float b = 2.0 * (ro.x * rd.x + ro.z * rd.z);
    float c = ro.x * ro.x + ro.z * ro.z - (radius * radius);

    float discriminant = b * b - 4.0 * a * c;

    if (discriminant < 0.0 || abs(a) < EPSILON) {
        return -1.0; // No intersection with infinite cylinder
    }
    
    float sqrtD = sqrt(discriminant);
    float t1 = (-b - sqrtD) / (2.0 * a);
    float t2 = (-b + sqrtD) / (2.0 * a);
    
    // Check y bounds for both intersections
    float validT = -1.0;
    
    // Check t1
    if (t1 > EPSILON) {
        float y1 = ro.y + t1 * rd.y;
        if (y1 >= -radius && y1 <= radius) {
            validT = t1;
        }
    }
    
    // Check t2 if t1 is not valid
    if (validT == -1.0 && t2 > EPSILON) {
        float y2 = ro.y + t2 * rd.y;
        if (y2 >= -radius && y2 <= radius) {
            validT = t2;
        }
    }
    
    // Check intersections with caps
    if (abs(rd.y) > EPSILON) {
        // Bottom cap (y = -0.5)
        float tBottom = (-radius - ro.y) / rd.y;
        if (tBottom > EPSILON && (validT == -1.0 || tBottom < validT)) {
            float x = ro.x + tBottom * rd.x;
            float z = ro.z + tBottom * rd.z;
            if (x * x + z * z <= 0.25) {
                validT = tBottom;
            }
        }
        
        // Top cap (y = 0.5)
        float tTop = (radius - ro.y) / rd.y;
        if (tTop > EPSILON && (validT == -1.0 || tTop < validT)) {
            float x = ro.x + tTop * rd.x;
            float z = ro.z + tTop * rd.z;
            if (x * x + z * z <= 0.25) {
                validT = tTop;
            }
        }
    }
    
    return validT;
}

// ----------------------------------------------
// normalCylinder: compute normal at intersection point in object space
vec3 normalCylinder(vec3 hitPos) {
    float radius = 0.5;
    vec3 normal = vec3(1.0);
    float py = hitPos.y;
    
    // Check if point is on a cap
    if (abs(py - radius) < EPSILON) {
        // Top cap
        normal.x = 0.0;
        normal.y = 1.0;
        normal.z = 0.0;
    } else if (abs(py + radius) < EPSILON) {
        // Bottom cap
        normal.x = 0.0;
        normal.y = -1.0;
        normal.z = 0.0;
    } else {
        // Side: normal is (x, 0, z) normalized
        normal.x = hitPos.x;
        normal.y = 0.0;
        normal.z = hitPos.z;
        normalize(normal);
    }
    
    return normal;
}

// ----------------------------------------------
// intersectCone: ray-cone intersection in object space
float intersectCone(vec3 ro, vec3 rd) {
    float t_min = 1e20;

    vec3 coneAxis = vec3(0.0, 1.0, 0.0); // y-axis
    vec3 coneApex = vec3(0.0, 0.5, 0.0); // apex at (0, 0.5, 0)

    // shift the origin to the apex
    ro = ro - coneApex;
    // the tan of the half angle between the axis and the cone's surface
    // radius = 0.5 over height = 1 gives us k = 0.5
    float k = 0.5;
    float k_sq = k * k;

    float A = dot(rd, rd) - (1.0 + k_sq) * pow(dot(rd, coneAxis), 2.0);
    float B = 2.0 * (dot(rd, ro) - (1.0 + k_sq) * dot(rd, coneAxis) * dot(ro, coneAxis));
    float C = dot(ro, ro) - (1.0 + k_sq) * pow(dot(ro, coneAxis), 2.0);

    float discriminant = (B * B) - (4.0 * A * C);
    if (discriminant > 0.0) {
        float t1 = (-B + sqrt(discriminant)) / (2.0 * A);
        float t2 = (-B - sqrt(discriminant)) / (2.0 * A);

        // calculate the intersection point and check if it falls within the unit cone's height
        // y = [-0.5, 0.5]
        if (t1 > EPSILON) {
            vec3 p_intersect = ro + rd * t1;
            float h = dot(p_intersect, coneAxis);
            if (h <= EPSILON && h >= -1.0 - EPSILON) {
                t_min = min(t_min, t1);
            }
        }
        if (t2 > EPSILON) {
            vec3 p_intersect = ro + rd * t2;
            float h = dot(p_intersect, coneAxis);
            if (h <= EPSILON && h >= -1.0 - EPSILON) {
                t_min = min(t_min, t2);
            }
        }
    }

    // calculate the cone cap
    vec3 capNorm = vec3(0.0, -1.0, 0.0);
    vec3 capOrigin = vec3(0.0, -0.5, 0.0);
    // shift back to original origin
    ro = ro + coneApex;

    float S = dot(capNorm, rd);
    if (abs(S) < EPSILON) return (t_min > EPSILON) ? t_min : -1.0;
    float Q = dot(capNorm, capOrigin);
    float R = dot(capNorm, ro);

    float t_cap = (Q - R) / S;
    if (t_cap <= EPSILON || t_cap >= t_min) return (t_min > EPSILON) ? t_min : -1.0;

    vec3 p_intersect = ro + rd * t_cap;

    if (p_intersect.x * p_intersect.x + p_intersect.z * p_intersect.z <= k_sq && t_cap > EPSILON) {
        t_min = min(t_cap, t_min);
    }
    return (t_min > EPSILON) ? t_min : -1.0;
}

// ----------------------------------------------
// normalCone: compute normal at intersection point in object space
vec3 normalCone(vec3 hitPos) {
    // cap normal (base at y = -0.5)
    // outward facing normal for the base disk is -y
    if (abs(hitPos.y + 0.5) <= EPSILON) return vec3(0, -1, 0);

    // side normal (use cone gradient)
    // N = (P - C) - (1 + k^2) * V * m
    // where V = (0, 1, 0), C = (0, 0.5, 0), m = (P - C) dot V = (y - 0.5)
    // For V = (0, 1, 0) this reduces to: [ x, -k^2 * (y - 0.5), z ]
    float k_sq = 0.25;
    float PCx = hitPos.x;
    float PCy = hitPos.y - 0.5;
    float PCz = hitPos.z;

    vec3 normal = vec3(PCx, -k_sq * PCy, PCz);
    
    // guard near the apex to avoid zero length normalization
    float len = sqrt(dot(normal, normal));
    if (len < EPSILON) {
        return vec3(0.0, 1.0, 0.0); // normal at apex points up
    }
    return normal / len;
}

vec2 getTexCoordSphere(vec3 hit, vec2 repeatUV) {
    vec3 n = normalize(hit);

    // Convert to spherical coordinates
    // u is the azimuthal angle (0 to 1 maps to 0 to 2pi)
    // v is the polar angle (0 to 1 maps to 0 to pi)
    float u = 0.5 + atan(n.z, n.x) / (2.0 * PI);
    float v = 0.5 - asin(n.y) / PI;

    // Apply repeat
    return vec2(u, v) * repeatUV;
}

vec2 getTexCoordCube(vec3 hit, vec3 dominantFace, vec2 repeatUV) {
    vec2 uv;

    // Determine what face we're on and project accordingly
    if (abs(dominantFace.x) > 0.5) {
        uv = vec2(hit.z, hit.y);
    } else if (abs(dominantFace.y) > 0.5) {
        uv = vec2(hit.x, hit.z);
    } else {
        uv = vec2(hit.x, hit.y);
    }

    uv = uv + 0.5;
    
    return uv * repeatUV;
}

vec2 getTexCoordCylinder(vec3 hit, vec2 repeatUV) {
   // For curved surface, u wraps the circumfrence while v follows y axis

   float u = 0.5 + atan(hit.z, hit.x) / (2.0 * PI);
   float v = hit.y + 0.5;
   
   return vec2(u, v) * repeatUV;
}

vec2 getTexCoordCone(vec3 hit, vec2 repeatUV) {
    // TODO: implement conical mapping
    return vec2(0.0);
}


// ----------------------------------------------
// getWorldRayDir: reconstruct world-space ray direction using uCamWorldMatrix
vec3 getWorldRayDir() {
    // get pixel in the [0, 1] range
    vec2 uv  = gl_FragCoord.xy / uResolution;

    vec2 ndc = uv * 2.0 - 1.0;
    vec4 camSpacePos = vec4(ndc, -1.0, 1.0); // assuming near plane at z = -1

    vec4 worldPos = uCamWorldMatrix * camSpacePos;
    
    vec3 dir = normalize(worldPos.xyz - uCameraPos);
    return dir;
}

// to help test occlusion (shadow)
bool isInShadow(vec3 shadowOrigin, vec3 lightDir, float maxDist, int ignoreIndex) {
    // Iterate through all objects in scene to test for shadows
    for (int i = 0; i < MAX_OBJECTS; i++) {
        if (i >= uObjectCount) break;
        if (i == ignoreIndex) continue; // skip self intersection

        // Get object transformations
        mat4 worldMatrix = fetchWorldMatrix(i);
        mat4 invWorldMatrix = inverse(worldMatrix);

        // Tranform shadow ray to object space
        vec3 roObj = (invWorldMatrix * vec4(shadowOrigin, 1.0)).xyz;
        vec3 rdObj = normalize((invWorldMatrix * vec4(lightDir, 0.0)).xyz);

        // Test intersection based on object type
        // * Dependent on scene buffer. Need to implement same way as in traceRay*
        float objectType = fetchFloat(0, i);
        float t = -1.0;
        
        if (objectType == 0.0) { // CUBE
            t = intersectCube(roObj, rdObj);
        }
        else if (objectType == 1.0) { // CYLIN
            t = intersectCylinder(roObj, rdObj);
        }
        else if (objectType == 2.0) { // CONE
            t = intersectCone(roObj, rdObj);
        }
        else if (objectType == 3.0) { // SPHERE
            t = intersectSphere(roObj, rdObj);
        }

        // If we hit something between surface and light, we're in shadow
        if (t > EPSILON) {
            // convert the intersection back to world space
            vec3 hitObj = roObj + t * rdObj;
            vec3 hitWorld = (worldMatrix * vec4(hitObj, 1.0)).xyz;
            float d = length(hitWorld - shadowOrigin);
            if (d > EPSILON && d < maxDist) {
                return true;
            }
        }
    }

    return false; 
}


// bounce = recursion level (0 for primary rays)
vec3 traceRay(vec3 ro, vec3 rayDir) {
    float closestD = 1e20;
    int hitIndex = -1;
    vec3 bestHitWorld  = vec3(0.0);
    vec3 bestNormalWorld = vec3(0.0);
    
    // Find closest intersection
    for (int i = 0; i < MAX_OBJECTS; ++i) {
        if (i >= uObjectCount) break;

        // Object-to-world matrix
        mat4 M = fetchWorldMatrix(i);
        // World-to-object
        mat4 invM = inverse(M);

        // Transform ray into object space
        vec3 roObj = (invM * vec4(ro, 1.0)).xyz;
        vec3 rdObj = normalize((invM * vec4(rayDir, 0.0)).xyz);

        // test intersection based on object type
        float objectType = fetchFloat(0, i);
        float t = 0.0;
        if (objectType == 0.0) {
            // cube
            t = intersectCube(roObj, rdObj);
        }
        else if (objectType == 1.0) {
            // cylinder
            t = intersectCylinder(roObj, rdObj);
        }
        else if (objectType == 2.0) {
            // cone
            t = intersectCone(roObj, rdObj);
        }
        else if (objectType == 3.0) {
            // sphere
            t = intersectSphere(roObj, rdObj);
        }
        
        if (t < EPSILON) continue;

        // convert the t from object space to world space
        // Compute hit position and normal
        vec3 hitObj = roObj + t * rdObj;   // object space
        vec3 hitWorld = (M * vec4(hitObj, 1.0)).xyz;

        mat3 normalMat = mat3(transpose(invM));
        vec3 normalObj = vec3(1.0);

        if (objectType == 0.0) {
            // cube
            normalObj = normalCube(hitObj);
        }
        else if (objectType == 1.0) {
            // cylinder
            normalObj = normalCylinder(hitObj);
        }
        else if (objectType == 2.0) {
            // cone
            normalObj = normalCone(hitObj);
        }
        else if (objectType == 3.0) {
            // sphere
            normalObj = normalSphere(hitObj);
        }

        vec3 normalWorld = normalize(normalMat * normalObj);

        float d = length(hitWorld - ro);

        if (d > EPSILON && d < closestD) {
            closestD = d;
            hitIndex = i;
            bestHitWorld = hitWorld;
            bestNormalWorld = normalWorld;
        }
    }

    // No hit -> set pixel same as the background color
    if (hitIndex < 0) return vec3(0.0);

    // Fetch material
    Material mat = fetchMaterial(hitIndex);

    vec3 hitWorld  = bestHitWorld;
    vec3 normalWorld = bestNormalWorld;

    // Phong shading, no shadows and reflections for now
    // ambient term
    vec3 color = uGlobalKa * mat.ambientColor;

    for (int li = 0; li < MAX_LIGHTS; ++li) {
        if (li >= uNumLights) break;

        int lightType = uLightType[li];
        vec3 L;
        float distToLight;
        vec3 shadowOrigin = hitWorld + normalWorld * EPSILON;

        if (lightType == 0) {
            // point light
            vec3 lightPos = uLightPos[li];
            // Avoid self-intersection w/ slight offset
            L = lightPos - shadowOrigin;
            distToLight = length(L);
            if (distToLight <= 0.0) continue;
            L /= distToLight; 
        }
        else {
            // directional light
            vec3 dir = normalize(uLightDir[li]);
            // directional lights are located at infinity with a constant direction
            L = -dir;
            distToLight = 1e20;
        }

        if (isInShadow(shadowOrigin, L, distToLight, hitIndex)) {
            continue;
        }

        // diffuse term
        float dotNL = max(dot(normalWorld, L), 0.0);
        vec3 diffuse = uGlobalKd * mat.diffuseColor * dotNL;

        // specular term
        vec3 V = normalize(uCameraPos - hitWorld);
        vec3 R = reflect(-L, normalWorld);
        float dotRV = max(dot(R, V), 0.0);

        // guard weird shininess values, got some weird issues with specular for
        // cube_test.xml and unit_cube.xml
        float shininess = clamp(mat.shininess, 1.0, 256.0);
        float sTerm = 0.0;
        if (dotRV > 0.0) sTerm = pow(dotRV, shininess);

        // float specStrength = uGlobalKs * ((mat.shininess + 2.0) * 0.5) * sTerm;
        float specStrength = uGlobalKs * sTerm;
        vec3 specular = mat.specularColor * specStrength;

        color += uLightColor[li] * (diffuse + specular);
    }

    // clamp to [0,1]
    color = clamp(color, vec3(0.0), vec3(1.0));
    return color;
}


// ----------------------------------------------
// main: iterate over all objects, test intersection, and shade
void main() {
    // Compute ray origin and direction in world space
    vec3 ro = uCameraPos;
    vec3 rayDir    = getWorldRayDir();

    // process and get final color 
    vec3 color = traceRay(ro, rayDir);
    outColor = vec4(color, 1.0);
}