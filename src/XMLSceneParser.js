import { SceneGlobalData, SceneCameraData, SceneLightData, SceneNode, SceneTransformation, ScenePrimitive, TransformationType, PrimitiveType, LightType, SceneFileMap } from './SceneDataStructures.js';
// Helper parsing functions

/**
 * Parse a <translate> or <rotate> or <scale> element: expects attributes x,y,z
 * Returns an array [x, y, z] (numbers).
 */
function parseTriple(elem) {
    const x = parseFloat(elem.getAttribute('x'));
    const y = parseFloat(elem.getAttribute('y'));
    const z = parseFloat(elem.getAttribute('z'));
    return [x, y, z];
}

/**
 * Parse a <float>‐like attribute on an element: assumes first attribute is the value.
 */
function parseFloatAttr(elem) {
    const val = parseFloat(elem.getAttribute(elem.attributes[0].name));
    return val;
}

/**
 * Parse a <color r="..." g="..." b="..." a="..."> element.
 * Return an object {r,g,b,a}. If 'a' is missing, default to 1.0.
 */
function parseColor(elem) {
    const attrs = elem.attributes;
    const r = parseFloat(elem.getAttribute('r'));
    const g = parseFloat(elem.getAttribute('g'));
    const b = parseFloat(elem.getAttribute('b'));
    let a = 1.0;
    if (elem.hasAttribute('a')) {
        a = parseFloat(elem.getAttribute('a'));
    }
    return { r, g, b, a };
}

/**
 * Parse a <matrix> element containing 4 row elements, each with 4 floats.
 * Returns a Float32Array of length 16 in row-major order.
 */
function parseMatrix(elem) {
    const mat = new Float32Array(16);
    // Each child of <matrix> is a <row> or similar; assume exactly 4 children
    const rows = elem.children;
    for (let row = 0; row < 4; row++) {
        const rElem = rows[row];
        const vals = [];
        for (let i = 0; i < rElem.attributes.length; i++) {
            vals.push(parseFloat(rElem.attributes[i].value));
        }
        // vals should have length 4
        for (let col = 0; col < 4; col++) {
            // row-major: index = row*4 + col
            mat[row * 4 + col] = vals[col];
        }
    }
    return mat;
}

/**
 * Parse a <texture filename="..." repeatU="..." repeatV="..."> element.
 * Returns a SceneFileMap object.
 */
function parseMap(elem) {
    const map = new SceneFileMap();
    map.filename = elem.getAttribute('filename') || elem.getAttribute('file');

    const uAttr = elem.getAttribute('repeatU') || elem.getAttribute('u');
    const vAttr = elem.getAttribute('repeatV') || elem.getAttribute('v');

    if (uAttr) {
        map.repeatU = parseFloat(uAttr);
    }
    if (vAttr) {
        map.repeatV = parseFloat(vAttr);
    }
    map.isUsed = true;
    return map;
}

// Main XMLSceneParser class

class XMLSceneParser {
    constructor() {
        this.globalData = new SceneGlobalData();
        this.cameraData = new SceneCameraData();
        this.lights = [];               // SceneLightData[]
        this.objects = {};               // name → SceneNode
        this.nodes = [];               // all nodes for cleanup if needed
    }

    /**
     * Parse the given XML string (scene description) and populate fields.
     * Returns a Promise that resolves to true/false.
     */
    async parseFromString(xmlString) {
        // 1. DOMParser to parse XML into a document
        const parser = new DOMParser();
        const xmlDoc = parser.parseFromString(xmlString, "application/xml");
        const root = xmlDoc.documentElement; // should be <scenefile>
        if (!root || root.tagName !== "scenefile") {
            console.error("Invalid XML: root is not <scenefile>");
            return false;
        }

        // 2. Initialize defaults (same as C++ did before looping)
        this._setDefaultCamera();
        this._setDefaultGlobalData();

        // 3. Loop over child nodes of <scenefile>
        const children = root.children;
        for (let i = 0; i < children.length; i++) {
            const elem = children[i];
            switch (elem.tagName) {
                case "globaldata":
                    if (!this._parseGlobalData(elem)) {
                        console.error("Failed to parse <globaldata>");
                        return false;
                    }
                    break;
                case "lightdata":
                    if (!this._parseLightData(elem)) {
                        console.error("Failed to parse <lightdata>");
                        return false;
                    }
                    break;
                case "cameradata":
                    if (!this._parseCameraData(elem)) {
                        console.error("Failed to parse <cameradata>");
                        return false;
                    }
                    break;
                case "object":
                    if (!this._parseObjectData(elem)) {
                        console.error("Failed to parse <object>");
                        return false;
                    }
                    break;
                case "reflective":
                    
                default:
                    console.warn(`Unsupported element <${elem.tagName}>`);
                    return false;
            }
        }

        return true;
    }

    /**
     * Set default camera values in case XML omits some fields.
     */
    _setDefaultCamera() {
        this.cameraData.isDir = false;
        this.cameraData.pos = [5, 5, 5];
        this.cameraData.up = [0, 1, 0];
        this.cameraData.lookAt = [0, 0, 0];
        this.cameraData.heightAngle = 45.0;
        this.cameraData.aspectRatio = 1.0;
        this.cameraData.aperture = 0.0;
        this.cameraData.focalLength = 1.0;
    }

    /**
     * Set default global data values.
     */
    _setDefaultGlobalData() {
        this.globalData.ka = 0.5;
        this.globalData.kd = 0.5;
        this.globalData.ks = 0.5;
        this.globalData.kt = 0.5;
    }

    /**
     * Parse <globaldata> children: <diffusecoeff>, <ambientcoeff>, <specularcoeff>, <transparentcoeff>
     */
    _parseGlobalData(elem) {
        const items = elem.children;
        for (let i = 0; i < items.length; i++) {
            const child = items[i];
            switch (child.tagName) {
                case "diffusecoeff":
                    this.globalData.kd = parseFloat(child.attributes[0].value);
                    break;
                case "ambientcoeff":
                    this.globalData.ka = parseFloat(child.attributes[0].value);
                    break;
                case "specularcoeff":
                    this.globalData.ks = parseFloat(child.attributes[0].value);
                    break;
                case "transparentcoeff":
                    this.globalData.kt = parseFloat(child.attributes[0].value);
                    break;
                default:
                    console.error(`Invalid globaldata type <${child.tagName}>`);
                    return false;
            }
        }
        return true;
    }

    /**
     * Parse <lightdata>: create a SceneLightData, read its children.
     */
    _parseLightData(elem) {
        const light = new SceneLightData();
        // Provide some defaults
        light.pos = [3, 3, 3];
        light.dir = [0, 0, 0];
        light.color.r = light.color.g = light.color.b = 1.0;
        light.function = [1, 0, 0];

        const items = elem.children;
        for (let i = 0; i < items.length; i++) {
            const child = items[i];
            switch (child.tagName) {
                case "id":
                    const val = child.getAttribute("value") || child.getAttribute("v");
                    light.id = parseInt(val);
                    // light.id = parseInt(child.getAttribute("value"));
                    break;
                case "type":
                    // const typeStr = child.getAttribute("value");
                    const typeStr = child.getAttribute("value") || child.getAttribute("id");
                    if (typeStr === "directional") {
                        light.type = LightType.LIGHT_DIRECTIONAL;
                    } else if (typeStr === "point") {
                        light.type = LightType.LIGHT_POINT;
                    } else if (typeStr === "spot") {
                        light.type = LightType.LIGHT_SPOT;
                    } else if (typeStr === "area") {
                        light.type = LightType.LIGHT_AREA;
                    } else {
                        console.error(`Unknown light type ${typeStr}`);
                        return false;
                    }
                    break;
                case "color":
                    Object.assign(light.color, parseColor(child));
                    break;
                case "function":
                    light.function = parseTriple(child);
                    break;
                case "position":
                    if (light.type === LightType.LIGHT_DIRECTIONAL) {
                        console.error("Position not applicable to directional lights");
                        return false;
                    }
                    light.pos = parseTriple(child);
                    break;
                case "direction":
                    if (light.type === LightType.LIGHT_POINT) {
                        console.error("Direction not applicable to point lights");
                        return false;
                    }
                    light.dir = parseTriple(child);
                    break;
                case "radius":
                    if (light.type !== LightType.LIGHT_SPOT) {
                        console.error("Radius only for spot lights");
                        return false;
                    }
                    light.radius = parseFloat(child.getAttribute("value"));
                    break;
                case "penumbra":
                    if (light.type !== LightType.LIGHT_SPOT) {
                        console.error("Penumbra only for spot lights");
                        return false;
                    }
                    light.penumbra = parseFloat(child.getAttribute("value"));
                    break;
                case "angle":
                    if (light.type !== LightType.LIGHT_SPOT) {
                        console.error("Angle only for spot lights");
                        return false;
                    }
                    // convert degrees to radians
                    light.angle = parseFloat(child.getAttribute("value")) * Math.PI / 180.0;
                    break;
                case "width":
                    if (light.type !== LightType.LIGHT_AREA) {
                        console.error("Width only for area lights");
                        return false;
                    }
                    light.width = parseFloat(child.getAttribute("value"));
                    break;
                case "height":
                    if (light.type !== LightType.LIGHT_AREA) {
                        console.error("Height only for area lights");
                        return false;
                    }
                    light.height = parseFloat(child.getAttribute("value"));
                    break;
                default:
                    console.warn(`Unknown element in <lightdata>: <${child.tagName}>`);
            }
        }

        this.lights.push(light);
        return true;
    }

    /**
     * Parse <cameradata>: children could be <pos>, <look>, <focus>, <up>, <heightangle>, <aspectratio>, <aperture>, <focallength>
     */
    _parseCameraData(elem) {
        const items = elem.children;
        let focusFound = false, lookFound = false;

        for (let i = 0; i < items.length; i++) {
            const child = items[i];
            switch (child.tagName) {
                case "pos":
                    this.cameraData.pos = parseTriple(child);
                    break;
                case "look":
                    this.cameraData.look = parseTriple(child);
                    this.cameraData.isDir = true;
                    lookFound = true;
                    break;
                case "focus":
                    this.cameraData.lookAt = parseTriple(child);
                    this.cameraData.isDir = false;
                    focusFound = true;
                    break;
                case "up":
                    this.cameraData.up = parseTriple(child);
                    break;
                case "heightangle":
                    this.cameraData.heightAngle = parseFloat(child.getAttribute("value"));
                    break;
                case "aspectratio":
                    this.cameraData.aspectRatio = parseFloat(child.getAttribute("value"));
                    break;
                case "aperture":
                    this.cameraData.aperture = parseFloat(child.getAttribute("value"));
                    break;
                case "focallength":
                    this.cameraData.focalLength = parseFloat(child.getAttribute("value"));
                    break;
                default:
                    console.warn(`Unknown element in <cameradata>: <${child.tagName}>`);
                    return false;
            }
        }

        if (focusFound && lookFound) {
            console.error("Camera cannot have both <look> and <focus>");
            return false;
        }

        return true;
    }

    /**
     * Parse <object type="tree" name="...">: create a SceneNode, register it, then process each <transblock>
     */
    _parseObjectData(elem) {
        const typeAttr = elem.getAttribute("type");
        if (typeAttr !== "tree") {
            console.error("Invalid object type (must be 'tree'):", typeAttr);
            return false;
        }
        const name = elem.getAttribute("name");
        if (!name) {
            console.error("<object> missing name attribute");
            return false;
        }
        if (this.objects[name]) {
            console.error(`Duplicate object name: ${name}`);
            return false;
        }

        // Create the node and register
        const node = new SceneNode();
        this.nodes.push(node);
        this.objects[name] = node;

        // Iterate over <transblock> children
        for (let i = 0; i < elem.children.length; i++) {
            const childElem = elem.children[i];
            if (childElem.tagName.toLowerCase() === "transblock") {
                const childNode = new SceneNode();
                this.nodes.push(childNode);
                if (!this._parseTransBlock(childElem, childNode)) {
                    console.error("Failed to parse <transblock> for object", name);
                    return false;
                }
                node.children.push(childNode);
            }
        }

        return true;
    }

    /**
     * Parse a <transblock> element into a given SceneNode.
     * Handles <translate>, <rotate>, <scale>, <matrix>, and nested <object> elements.
     */
    _parseTransBlock(transblockElem, node) {
        const items = transblockElem.children;
        for (let i = 0; i < items.length; i++) {
            const child = items[i];
            switch (child.tagName) {
                case "translate": {
                    const t = new SceneTransformation();
                    t.type = TransformationType.TRANSFORMATION_TRANSLATE;
                    t.translate = parseTriple(child);
                    node.transformations.push(t);
                    break;
                }
                case "rotate": {
                    const t = new SceneTransformation();
                    t.type = TransformationType.TRANSFORMATION_ROTATE;
                    const vals = parseTriple(child);
                    t.rotate = vals;
                    const ang = child.getAttribute("angle");
                    if (ang !== null) {
                        t.angle = parseFloat(ang) * Math.PI / 180.0; // degrees→radians
                    }
                    node.transformations.push(t);
                    break;
                }
                case "scale": {
                    const t = new SceneTransformation();
                    t.type = TransformationType.TRANSFORMATION_SCALE;
                    t.scale = parseTriple(child);
                    node.transformations.push(t);
                    break;
                }
                case "matrix": {
                    const t = new SceneTransformation();
                    t.type = TransformationType.TRANSFORMATION_MATRIX;
                    t.matrix = parseMatrix(child);
                    node.transformations.push(t);
                    break;
                }
                case "object": {
                    const objType = child.getAttribute("type");
                    if (objType === "master") {
                        const masterName = child.getAttribute("name");
                        const masterNode = this.objects[masterName];
                        if (!masterNode) {
                            console.error("Invalid master object reference:", masterName);
                            return false;
                        }
                        node.children.push(masterNode);
                    }
                    else if (objType === "tree") {
                        for (let i = 0; i < child.children.length; i++) {
                            const subElem = child.children[i];
                            if (subElem.tagName.toLowerCase() === "transblock") {
                                const subNode = new SceneNode();
                                this.nodes.push(subNode);
                                if (!this._parseTransBlock(subElem, subNode)) {
                                    console.error(
                                        "Failed to parse nested <transblock> in <object type='tree'>"
                                    );
                                    return false;
                                }
                                node.children.push(subNode);
                            }
                        }
                    }
                    else if (objType === "primitive") {
                        if (!this._parsePrimitive(child, node)) {
                            console.error("Failed to parse <primitive>");
                            return false;
                        }
                    }
                    else {
                        console.error("Unknown <object> type:", objType);
                        return false;
                    }
                    break;
                }
                default:
                    console.error("Invalid transblock element type:", child.tagName);
                    return false;
            }
        }
        return true;
    }

    /**
     * Parse a <primitive> element and append a ScenePrimitive to node.primitives.
     * Supports <sphere>, <cube>, <cylinder>, <cone>, <mesh>, and material properties.
     */
    _parsePrimitive(primElem, node) {
        const primitive = new ScenePrimitive();

        // Determine primitive type from first attribute or child tag
        const firstAttrVal = primElem.attributes.name.value;
        switch (firstAttrVal) {
            case "sphere":
                primitive.type = PrimitiveType.SHAPE_SPHERE;
                break;
            case "cube":
                primitive.type = PrimitiveType.SHAPE_CUBE;
                break;
            case "cylinder":
                primitive.type = PrimitiveType.SHAPE_CYLINDER;
                break;
            case "cone":
                primitive.type = PrimitiveType.SHAPE_CONE;
                break;
            case "mesh":
                primitive.type = PrimitiveType.SHAPE_MESH;
                // Next attribute is mesh filename
                const meshAttr = primElem.attributes[1];
                if (!meshAttr) {
                    console.error("Mesh primitive must specify mesh file");
                    return false;
                }
                primitive.meshfile = meshAttr.value;
                break;
            default:
                console.error("Unknown primitive type:", firstAttrVal);
                return false;
        }

        // Initialize default material
        const mat = primitive.material;
        mat.textureMap.isUsed = false;
        mat.bumpMap.isUsed = false;
        mat.cDiffuse = { r: 1, g: 1, b: 1, a: 1 };

        // Parse material sub-elements
        const items = primElem.children;
        for (let i = 0; i < items.length; i++) {
            const child = items[i];
            switch (child.tagName) {
                case "diffuse":
                    mat.cDiffuse = parseColor(child);
                    break;
                case "ambient":
                    mat.cAmbient = parseColor(child);
                    break;
                case "reflective":
                    mat.cReflective = parseColor(child);
                    break;
                case "specular":
                    mat.cSpecular = parseColor(child);
                    break;
                case "emissive":
                    mat.cEmissive = parseColor(child);
                    break;
                case "transparent":
                    mat.cTransparent = parseColor(child);
                    break;
                case "shininess":
                    mat.shininess = parseFloat(child.getAttribute("value"));
                    break;
                case "ior":
                    mat.ior = parseFloat(child.getAttribute("value"));
                    break;
                case "texture":
                    mat.textureMap = parseMap(child);
                    break;
                case "bumpmap":
                    mat.bumpMap = parseMap(child);
                    break;
                case "blend":
                    mat.blend = parseFloat(child.getAttribute("value"));
                    break;
                default:
                    console.warn(`Unknown primitive data <${child.tagName}>`);
                    return false;
            }
        }

        node.primitives.push(primitive);
        return true;
    }

    /**
     * Helper to retrieve the root node (name === "root"), or null if not found.
     */
    getRootNode() {
        return this.objects["root"] || null;
    }

    /**
     * Retrieve the list of lights.
     */
    getLights() {
        return this.lights.slice();  // return a copy
    }

    /**
     * Retrieve camera data.
     */
    getCameraData() {
        return Object.assign({}, this.cameraData);
    }

    /**
     * Retrieve global data.
     */
    getGlobalData() {
        return Object.assign({}, this.globalData);
    }
}

export { XMLSceneParser }; 