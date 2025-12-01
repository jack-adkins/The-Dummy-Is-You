const LightType = {
    LIGHT_POINT:        0,
    LIGHT_DIRECTIONAL:  1,
    LIGHT_SPOT:         2,
    LIGHT_AREA:         3
};

const TransformationType = {
    TRANSFORMATION_TRANSLATE: 0,
    TRANSFORMATION_SCALE:     1,
    TRANSFORMATION_ROTATE:    2,
    TRANSFORMATION_MATRIX:    3
};

const PrimitiveType = {
    SHAPE_CUBE:     0,
    SHAPE_CYLINDER: 1,
    SHAPE_CONE:     2,
    SHAPE_SPHERE:   3,
    SHAPE_SPECIAL1: 4,
    SHAPE_SPECIAL2: 5,
    SHAPE_SPECIAL3: 6,
    SHAPE_MESH:     7
};

// Data structures

class SceneGlobalData {
    constructor() {
        this.ka = 0.0;  // ambient coefficient
        this.kd = 0.0;  // diffuse coefficient
        this.ks = 0.0;  // specular coefficient
        this.kt = 0.0;  // transparency coefficient
    }
}

class SceneCameraData {
    constructor() {
        this.pos =      [0, 0, 0];
        this.lookAt =   [0, 0, 0];
        this.look =     [0, 0, 0];
        this.up =       [0, 1, 0];
        this.isDir =    true;          // true: use look vector; false: use lookAt point
        this.heightAngle = 45.0;       // in degrees
        this.aspectRatio = 1.0;
        this.aperture =    0.0;        // for depth of field
        this.focalLength = 1.0;        // for depth of field
    }
}

class SceneFileMap {
    constructor() {
        this.isUsed = false;
        this.filename = "";
        this.repeatU = 1.0;
        this.repeatV = 1.0;
        this.textureIndex = 0; // new for a4 since files reference texture mapping 
    }
}

class SceneMaterial {
    constructor() {
        this.cDiffuse = { r: 1, g: 1, b: 1, a: 1 };
        this.cAmbient = { r: 0, g: 0, b: 0, a: 1 };
        this.cReflective = { r: 0, g: 0, b: 0, a: 1 };
        this.cSpecular = { r: 0, g: 0, b: 0, a: 1 };
        this.cTransparent = { r: 0, g: 0, b: 0, a: 1 };
        this.cEmissive = { r: 0, g: 0, b: 0, a: 1 };
        this.textureMap = new SceneFileMap();
        this.bumpMap = new SceneFileMap();
        this.blend = 0.0;
        this.shininess = 0.0;
        this.ior = 1.0;  // index of refraction
    }
}

class ScenePrimitive {
    constructor() {
        this.type = PrimitiveType.SHAPE_CUBE;
        this.meshfile = "";
        this.material = new SceneMaterial();
    }
}

class SceneTransformation {
    constructor() {
        this.type = TransformationType.TRANSFORMATION_TRANSLATE;
        this.translate = [0, 0, 0];
        this.scale = [1, 1, 1];
        this.rotate = [0, 0, 1];
        this.angle = 0;                     // radians
        this.matrix = new Float32Array(16); // row-major 4×4
    }
}

class SceneNode {
    constructor() {
        this.transformations = []; // SceneTransformation[]
        this.primitives = [];      // ScenePrimitive[]
        this.children = [];        // SceneNode[]
    }
}

class SceneLightData {
    constructor() {
        this.id = 0;
        this.type = LightType.LIGHT_POINT;
        this.color = { r: 1, g: 1, b: 1, a: 1 };
        this.function = [1, 0, 0];
        this.pos = [0, 0, 0];
        this.dir = [0, 0, -1];
        this.radius = 0.0;      // for spot lights
        this.penumbra = 0.0;    // for spot lights
        this.angle = 0.0;       // for spot lights (radians)
        this.width = 0.0;       // for area lights
        this.height = 0.0;      // for area lights
    }
}

export {
    LightType,
    TransformationType,
    PrimitiveType,
    SceneGlobalData,
    SceneCameraData,
    SceneFileMap,
    SceneMaterial,
    ScenePrimitive,
    SceneTransformation,
    SceneNode,
    SceneLightData
};