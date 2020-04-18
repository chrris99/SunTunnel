//=============================================================================================
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Conforti Christian
// Neptun : F8R430
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================

#include "framework.h"

#pragma region Constants

const float EPSILON = 0.0001f;
const int MAX_DEPTH = 5;

#pragma endregion

#pragma region Operators

vec3 operator/(const vec3& num, const vec3& denom) {
	return vec3{ num.x / denom.x, num.y / denom.y, num.z / denom.z };
}

mat4 transpose(const mat4& m) {
	mat4 mTranspose;
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			mTranspose[i][j] = m[j][i];
		}
	}

	return mTranspose;
}

mat4 InverseTranslateMatrix(const vec3& t) {
	return mat4{ 
		vec4{ 1, 0, 0, 0 },
		vec4{ 0, 1, 0, 0 },
		vec4{ 0, 0, 1, 0 },
		vec4{ -t.x, -t.y, -t.z, 1 } 
	};
}

#pragma endregion

#pragma region Shaders

// Vertex Shader in GLSL
const char* vertexSource = R"(
	#version 330
	precision highp float;
	layout(location = 0) in vec2 cVertexPosition;
	out vec2 texcoord;
	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1);
	}
)";

// Fragment Shader in GLSL
const char* fragmentSource = R"(
	#version 330
	precision highp float;
	uniform sampler2D textureUnit;
	in vec2 texcoord;
	out vec4 fragmentColor;
	void main() {
		fragmentColor = texture(textureUnit, texcoord);	
	}
)";

#pragma endregion

#pragma region Material

enum MaterialType { ROUGH, REFLECTIVE};

struct Material {
	MaterialType type;	// ROUGH or REFLECTIVE
	vec3 ka, kd, ks;
	vec3 F0;
	float shine;

	Material(MaterialType t) { type = t; }
};

struct RoughMaterial : Material {
	RoughMaterial(vec3 _kd, vec3 _ks, float _shine) : Material(ROUGH) {
		ka = _kd * M_PI;
		kd = _kd;
		ks = _ks;
		shine = _shine;
	}
};

struct ReflectiveMaterial : Material {
	ReflectiveMaterial(vec3 n, vec3 kappa) : Material(REFLECTIVE) {
		vec3 one = { 1, 1, 1 };
		F0 = ((n - one) * (n - one) + kappa * kappa) / ((n + one) * (n + one) + kappa * kappa);
	}
};

// Pre-defined reflective and rough materials
namespace Materials {
	ReflectiveMaterial* GOLD() {
		return new ReflectiveMaterial(vec3{ 0.17f, 0.35f, 1.5f }, vec3{ 3.1f, 2.7f, 1.9f });
	}

	ReflectiveMaterial* SILVER() {
		return new ReflectiveMaterial(vec3{ 0.14f, 0.16f, 0.13f }, vec3{ 4.1f, 2.3f, 3.1f });
	}

	ReflectiveMaterial* COPPER() {
		return new ReflectiveMaterial(vec3{ 0.24f, 1.15f, 1.24f }, vec3{ 3.56f, 2.6f, 2.37f });
	}

	ReflectiveMaterial* ALUMINIUM() {
		return new ReflectiveMaterial(vec3{ 1.53f, 0.91f, 0.61f }, vec3{ 7.66f, 6.35f, 5.38f });
	}

	RoughMaterial* BROWN(const float& shine = 50) {
		return new RoughMaterial(vec3{ 0.3f, 0.2f, 0.1f }, vec3{ 1, 1, 1 }, shine);
	}

	RoughMaterial* LIGHTGREEN(const float& shine = 50) {
		return new RoughMaterial(vec3{ 0.2f, 0.5f, 0.3f }, vec3{ 1, 1, 1 }, shine);
	}

	RoughMaterial* BLUE(const float& shine = 50) {
		return new RoughMaterial(vec3{ 0.2f, 0.2f, 0.6f }, vec3{ 1, 1, 1 }, shine);
	}
}

#pragma endregion

#pragma region Light

/*
* Represents a light beam
*/
struct Ray {
	vec3 start, dir;
	Ray(const vec3& _start, const vec3& _dir) {
		start = _start;
		dir = normalize(_dir);
	}
};

/*
* Represents an object-ray intersection
*/
struct Hit {
	float t;	// Ray parameter (ray(t) = start + dir * t)
	vec3 position, normal;
	Material* material;

	/*
	* Default constructor sets the ray parameter to -1,
	* meaning that there was no (valuable) intersection
	*/

	Hit() : t(-1) { }
};

struct Light {
	vec3 direction, Le;

	Light(vec3 _direction, vec3 _Le) {
		direction = normalize(_direction);
		Le = _Le;
	}
};

#pragma endregion

#pragma region Objects

class QuadricIntersectable {
protected:
	mat4 Q;					// Symmetric matrix, representing a quadric object (transformed)
	vec3 cut;
	Material* material;		// Material of the quadric object
	Material* texture;		// Optional texture of the quadric object
	
	const mat4& translate(const mat4& m, const vec3& t) {
		mat4 invTransM = InverseTranslateMatrix(t);
		return invTransM * m * transpose(invTransM);
	}

	vec3 gradf(vec4 r) {
		vec4 g = r * Q * 2.0f;
		return normalize(vec3{ g.x, g.y, g.z });
	}

public:
	Hit intersect(const Ray& ray) {
		Hit hit; // Ray parameter t = -1

		vec4 S = { ray.start.x, ray.start.y, ray.start.z, 1 };
		vec4 D = { ray.dir.x, ray.dir.y, ray.dir.z, 0 };

		float a = dot(D * Q, D);
		float b = dot(D * Q, S) + dot(S * Q, D);
		float c = dot(S * Q, S);

		float discriminant = b * b - 4.0f * a * c;
		if (discriminant < 0) return hit;

		float t1 = (-b + sqrtf(discriminant)) / (2.0f * a);
		float t2 = (-b - sqrtf(discriminant)) / (2.0f * a);

		if (t1 <= 0) return hit;

		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t; // ray(t) = start + dir * t

		// If we want to cut the object
		if (cut.x == 1) {
			if (hit.position.z < cut.y || hit.position.z > cut.z) {
				hit.t = t1;
				hit.position = ray.start + ray.dir * hit.t;
				if (hit.position.z < cut.y || hit.position.z > cut.z) {
					hit.t = -1;  return hit;
				}
			}
		}


		hit.normal = (gradf(vec4{ hit.position.x, hit.position.y, hit.position.z, 1 }));
		hit.material = material;
		if (texture) { // texturing
			double u = acos(hit.normal.y) / M_PI;
			double v = (atan2(hit.normal.z, hit.normal.x) / M_PI + 1) / 2;
			int U = (int)(u * 6), V = (int)(v * 8);
			if (U % 2 ^ V % 2) hit.material = texture;
		}

		return hit;
	}
};

class Sphere final : public QuadricIntersectable {
public:
	Sphere(const vec3& _center, float radius, const vec3& _cut, Material* _material, Material* _texture = nullptr) {
		
		material = _material;
		texture = _texture;
		cut = _cut;

		mat4 untransformedQ = {
			{1 / (radius * radius), 0, 0, 0},
			{0, 1 / (radius * radius), 0, 0},
			{0, 0, 1 / (radius * radius), 0},
			{0, 0, 0, -1}
		};

		this->Q = translate(untransformedQ, _center);
	}
};

class Ellipsoid final : public QuadricIntersectable {
public:
	Ellipsoid(const vec3& center, const vec3& params, const vec3& _cut, Material* _material, Material* _texture = nullptr) {

		material = _material;
		texture = _texture;
		cut = _cut;

		mat4 untransformedQ = { 
			{1 / (params.x * params.x), 0, 0, 0},
			{0, 1 / (params.y * params.y), 0, 0},
			{0, 0, 1 / (params.z * params.z), 0},
			{0, 0, 0, -1}
		};

		this->Q = translate(untransformedQ, center);
	}


};

class Cylinder final : public QuadricIntersectable {
public:
	Cylinder(Material* material) {

	}
};
//const float& fromY, const float& toY =
class Hyperboloid : public QuadricIntersectable {
public:
	Hyperboloid(const vec3& center, const vec3& params, const vec3& _cut, Material* _material) {
		
		material = _material;
		cut = _cut;

		mat4 untransformedQ = {
			{1 / (params.x * params.x), 0, 0, 0},
			{0, 1 / (params.y * params.y), 0, 0},
			{0, 0, -1 / (params.z * params.z), 0},
			{0, 0, 0, -1}
		};

		this->Q = translate(untransformedQ, center);
	}
};

class Paraboloid final : public QuadricIntersectable {
public:
	Paraboloid(const vec3& center, const vec3& params, const vec3& _cut, Material* _material) {

		material = _material;
		cut = _cut;
		
		mat4 untransformedQ = {
			{ 1 / params.x, 0, 0, 0 },
			{ 0, 1 / params.y, 0, 0 },
			{ 0, 0, 0, params.z / 2 },
			{ 0, 0, params.z / 2, 0 }
		};

		Q = translate(untransformedQ, center);
	}
};
/*
class Cylinder : public QuadricIntersectable {
public:
	Cylinder(const vec3& center, const vec3& params, Material* _material) {

		material = _material;

		mat4 untransformedQ = {
			{1 / (params.x * params.x), 0, 0, 0},
			{ 0, 0, 0, 0 },
			{ 0, 0, 1 / (params.z * params.z), 0 },
			{ 0, 0, 0, -1 } 
		};

		this->Q = transform(untransformedQ, center);
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		Hit nonhit;

		vec4 D = vec4(ray.getDir().x, ray.getDir().y, ray.getDir().z, 0);
		vec4 S = vec4(ray.getStart().x, ray.getStart().y, ray.getStart().z, 1);
		float a = dot(D * Q, D);
		float b = dot(D * Q, S) + dot(S * Q, D);
		float c = dot(S * Q, S);

		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return nonhit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;    // t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return nonhit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.setPosition(ray.getStart() + ray.getDir() * hit.t);
		if (hit.getPosition().y > 0.2f || hit.getPosition().y < -0.5f) {
			hit.t = t1;
			hit.setPosition(ray.getStart() + ray.getDir() * hit.t);
			if (hit.getPosition().y > 0.2f || hit.getPosition().y < -0.5f) return nonhit;
		}
		hit.setNormal(gradf(vec4(hit.getPosition().x, hit.getPosition().y, hit.getPosition().z, 1)));
		hit.setMaterial(material);
		return hit;
	}
};

class OneSheetHyperbolid : public QuadricIntersectable {
public:
	OneSheetHyperbolid(const vec3& center, const vec3& params, Material* _material) {
		
		material = _material;

		mat4 untransformedQ = {
			{1 / (params.x * params.x), 0, 0, 0},
			{0, 1 / (params.y * params.y), 0, 0},
			{0, 0, 1 / (params.z * params.z), 0},
			{0, 0, 0, -1}
		};

		this->Q = transform(untransformedQ, center);
	}

	Hit intersect(const Ray& ray) override {
		Hit hit;
		Hit nonhit;

		vec4 D = vec4(ray.getDir().x, ray.getDir().y, ray.getDir().z, 0);
		vec4 S = vec4(ray.getStart().x, ray.getStart().y, ray.getStart().z, 1);
		float a = dot(D * Q, D);
		float b = dot(D * Q, S) + dot(S * Q, D);
		float c = dot(S * Q, S);

		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return nonhit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;    // t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return nonhit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.setPosition(ray.getStart() + ray.getDir() * hit.t);
		if (hit.getPosition().y > 0.2f || hit.getPosition().y < -0.5f) {
			hit.t = t1;
			hit.setPosition(ray.getStart() + ray.getDir() * hit.t);
			if (hit.getPosition().y > 0.2f || hit.getPosition().y < -0.5f) return nonhit;
		}
		hit.setNormal(gradf(vec4(hit.getPosition().x, hit.getPosition().y, hit.getPosition().z, 1)));
		hit.setMaterial(material);
		return hit;
	}
};
*/
#pragma endregion

#pragma region Camera and Scene

class Camera {
private:
	vec3 eye, lookat, right, up;

public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov) {
		eye = _eye;
		lookat = _lookat;

		vec3 w = eye - lookat;
		right = normalize(cross(vup, w)) * length(w) * tanf(fov / 2);
		up = normalize(cross(w, right)) * length(w) * tanf(fov / 2);
	}

	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
};

class Scene {
private:
	std::vector<QuadricIntersectable*> objects;
	std::vector<Light*> lights;
	Camera camera;
	vec3 ambientLight;


	Hit firstIntersect(const Ray& ray) {
		Hit bestHit;
		for (auto object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(const Ray& ray) {	// for directional lights
		for (auto object : objects) if (object->intersect(ray).t > 0) return true;
		return false;
	}

	vec3 trace(const Ray& ray, int depth = 0) {

		if (depth > MAX_DEPTH) return ambientLight;

		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return ambientLight;
		vec3 outRadiance = { 0, 0, 0 };

		if (hit.material->type == ROUGH) {
			outRadiance = hit.material->ka * ambientLight;
			for (auto light : lights) {
				Ray shadowRay(hit.position + hit.normal * EPSILON, light->direction);
				float cosTheta = dot(hit.normal, light->direction);
				if (cosTheta > 0 && !shadowIntersect(shadowRay)) {
					outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
					vec3 halfway = normalize(-ray.dir + light->direction);
					float cosDelta = dot(hit.normal, halfway);
					if (cosDelta > 0) outRadiance = outRadiance + light->Le * hit.material->ks * powf(cosDelta, hit.material->shine);
				}
			}
		}

		if (hit.material->type == REFLECTIVE) {
			vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
			// Bel�p�si sz�g : �rkez�si ir�ny �s a fel�leti norm�lis k�z�tti sz�g 
			float cosa = -dot(ray.dir, hit.normal);
			vec3 one = { 1,1,1 };
			vec3 Fresnel = hit.material->F0 + (one - hit.material->F0) * pow(1 - cosa, 5);
			outRadiance = outRadiance + trace(Ray(hit.position + hit.normal * EPSILON, reflectedDir), depth + 1) * Fresnel;
		}

		return outRadiance;
	}

public:
	void build() {
		// Create camera
		vec3 eye = vec3{ -1.8, 0, 0 };
		vec3 vup = vec3{ 0, 0, 1 };
		vec3 lookat = vec3{ 0, 0, 0 };
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		// Create lights
		ambientLight = vec3{ 0.4f, 0.4f, 0.4f };
		lights.push_back(new Light(vec3(-1.8, 0, 0), vec3(1, 1, 1)));



		// Create objects
		objects.push_back(new Sphere(vec3(1, 0, 0.1), 0.2, vec3(0, 0, 0), Materials::GOLD()));
		objects.push_back(new Sphere(vec3(2, 0.7, -0.5), 0.7, vec3(0, 0, 0), Materials::LIGHTGREEN()));

		//objects.push_back(new Ellipsoid(vec3(0, 0, 0), vec3(2, 2, 1), vec3(1, -1.0, 0.95), Materials::BROWN()));
		objects.push_back(new Ellipsoid(vec3(0.7, -0.2, -0.6), vec3(0.2, 0.2, 0.4), vec3(1, -0.9, -0.2), Materials::ALUMINIUM()));
		objects.push_back(new Ellipsoid(vec3(0.7, 0.3, -0.6), vec3(0.2, 0.2, 0.3), vec3(1, -0.7, -0.35), Materials::BLUE()));
	}

	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}
};

#pragma endregion

GPUProgram gpuProgram;
Scene scene;

class FullScreenTexturedQuad {
private:
	unsigned int vao; // Vertex Array Object ID
	Texture texture;  // Texture ID

public:
	FullScreenTexturedQuad(const int windowWidth, const int windowHeight, const std::vector<vec4>& image) : texture(windowWidth, windowHeight, image) {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		unsigned int vbo;
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);

		float vertexCoords[] = { -1, -1, 1, -1, 1, 1, -1, 1 };
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, reinterpret_cast<void*>(0));
	}

	void draw() {
		glBindVertexArray(vao);
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	}
};

FullScreenTexturedQuad* fullScreenTextureQuad;

#pragma region EventHandling

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	
	scene.build();
	
	std::vector<vec4> image(windowWidth * windowHeight);
	long timeStart = glutGet(GLUT_ELAPSED_TIME);
	scene.render(image);
	long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));

	fullScreenTextureQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	fullScreenTextureQuad->draw();
	glutSwapBuffers();												// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {

	glutPostRedisplay();
}

#pragma endregion