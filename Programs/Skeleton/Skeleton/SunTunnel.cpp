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

const float  PI_F = 3.14159265358979f;
const float EPSILON = 0.0001f;
const int MAX_DEPTH = 5;

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

#pragma region Operators

vec3 operator/(const vec3& num, const vec3& denom) {
	return vec3{ num.x / denom.x, num.y / denom.y, num.z / denom.z };
}

float distance(const vec3& p1, const vec3& p2) {
	return sqrtf((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z));
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



#pragma region Materials

enum class MaterialType { Rough, Reflective };

struct Material {
	MaterialType type;	// ROUGH or REFLECTIVE
	vec3 ka, kd, ks;
	vec3 F0;
	float shine;

	Material(const MaterialType& t) { type = t; }
};

struct RoughMaterial : Material {
	RoughMaterial(const vec3& _kd, const vec3& _ks = { 1.0f, 1.0f, 1.0f }, const float& _shine = 50.0f) : Material{ MaterialType::Rough } {
		ka = _kd * PI_F;
		kd = _kd;
		ks = _ks;
		shine = _shine;
	}
};

struct ReflectiveMaterial : Material {
	ReflectiveMaterial(const vec3& n, const vec3& kappa) : Material{ MaterialType::Reflective } {
		vec3 one = { 1.0f, 1.0f, 1.0f };
		F0 = ((n - one) * (n - one) + kappa * kappa) / ((n + one) * (n + one) + kappa * kappa);
	}
};

#pragma endregion

#pragma region Light

struct Ray {
	vec3 start, dir;
	Ray(const vec3& _start, const vec3& _dir) {
		start = _start;
		dir = normalize(_dir);
	}
};

struct Hit {
	float t;	// Ray parameter (ray(t) = start + dir * t)
	vec3 position, normal;
	Material* material;

	/*
	* Default constructor sets the ray parameter to -1,
	* meaning that there was no (valuable) intersection
	*/

	Hit() : t{ -1.0f } { }
};

struct Light {
	vec3 direction, Le;

	Light(const vec3& _direction, const vec3& _Le) {
		direction = normalize(_direction);
		Le = _Le;
	}
};

#pragma endregion

#pragma region Objects

class QuadricIntersectable {
protected:
	Material* material;
	mat4 Q;					// Symmetric matrix, representing a quadric object (transformed)
	vec3 cut;				// cut.x (0 or 1) - do we want to cut?, cut.y - from z, cut.z - to z

	const mat4 translate(const mat4& m, const vec3& t) {
		mat4 invTransM = InverseTranslateMatrix(t);
		return invTransM * m * transpose(invTransM);
	}

	vec3 gradf(const vec4& r) {
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

		return hit;
	}
};

class Ellipsoid final : public QuadricIntersectable {
public:
	Ellipsoid(const vec3& center, const vec3& params, Material* _material, const vec3& _cut = { 0.0f, 0.0f, 0.0f }) {

		material = _material;
		cut = _cut;

		mat4 untransformedQ = {
			{ 1 / (params.x * params.x), 0, 0, 0 },
			{ 0, 1 / (params.y * params.y), 0, 0 },
			{ 0, 0, 1 / (params.z * params.z), 0 },
			{ 0, 0, 0, -1 }
		};

		this->Q = translate(untransformedQ, center);
	}
};

class Hyperboloid final : public QuadricIntersectable {
public:
	Hyperboloid(const vec3& center, const vec3& params, Material* _material, const vec3& _cut = { 0.0f, 0.0f, 0.0f }) {

		material = _material;
		cut = _cut;

		mat4 untransformedQ = {
			{ 1 / (params.x * params.x), 0, 0, 0 },
			{ 0, 1 / (params.y * params.y), 0, 0 },
			{ 0, 0, -1 / (params.z * params.z), 0 },
			{ 0, 0, 0, -1 }
		};

		this->Q = translate(untransformedQ, center);
	}
};

class Paraboloid final : public QuadricIntersectable {
public:
	Paraboloid(const vec3& center, const vec3& params, Material* _material, const vec3& _cut = { 0.0f, 0.0f, 0.0f }) {

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

#pragma endregion

#pragma region Camera

class Camera {
private:
	vec3 eye, lookat, right, up;

public:
	void set(const vec3& _eye, const vec3& _lookat, const vec3& vup, const float& fov) {
		eye = _eye;
		lookat = _lookat;

		vec3 w = eye - lookat;
		right = normalize(cross(vup, w)) * length(w) * tanf(fov / 2);
		up = normalize(cross(w, right)) * length(w) * tanf(fov / 2);
	}

	Ray getRay(const int& X, const int& Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
};

#pragma endregion

#pragma region Scene

class Scene {
private:
	Camera camera;									// Camera recording the virtual world

	std::vector<QuadricIntersectable*> objects;			// Objects in the virtual world

	std::vector<Light*> lights;						// Lights in the virtual world (sunlight)
	vec3 ambientLight;								// Skylight
	std::vector<vec3> lightSourceSamples;			// Point samples on the sun tunnel, representing the light sources

	Hit firstIntersect(const Ray& ray) {
		Hit bestHit;
		for (QuadricIntersectable* object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(const Ray& ray) {	// for directional lights
		for (auto object = objects.begin() + 1; object != objects.end(); object++) {
			if ((*object)->intersect(ray).t > 0) return true;
		}
		return false;
	}

	vec3 trace(const Ray& ray, int depth = 0) {
		if (depth > MAX_DEPTH) return ambientLight;

		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return ambientLight + lights[0]->Le * powf(dot(ray.dir, lights[0]->direction), 10);
		vec3 outRadiance = { 0.0f, 0.0f, 0.0f };

		if (hit.material->type == MaterialType::Rough) {
			outRadiance = hit.material->ka * ambientLight;
			for (auto samplePoint : lightSourceSamples) {
				vec3 rayDir = samplePoint - hit.position;
				Ray shadowRay(hit.position + hit.normal * EPSILON, rayDir);
				float cosTheta = dot(hit.normal, rayDir);
				if (cosTheta > 0 && !shadowIntersect(shadowRay)) {
					float cosa = dot(normalize(vec3{ 0.0f, 0.0f, 0.0f } - samplePoint), normalize(hit.position - samplePoint));
					float omega = (0.625f * PI_F / lightSourceSamples.size()) * (cosa / powf(distance(hit.position, samplePoint), 2));
					vec3 Le = trace(Ray(hit.position + hit.normal * EPSILON, rayDir), depth + 1);
					outRadiance = outRadiance + Le * hit.material->kd * cosTheta * omega;
					vec3 halfway = normalize(-ray.dir + rayDir);
					float cosDelta = dot(hit.normal, halfway);
					if (cosDelta > 0) outRadiance = outRadiance + Le * hit.material->ks * powf(cosDelta, hit.material->shine) * omega;
				}
			}
		}

		if (hit.material->type == MaterialType::Reflective) {
			vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
			float cosa = -dot(ray.dir, hit.normal);
			vec3 one = { 1.0f, 1.0f, 1.0f };
			vec3 Fresnel = hit.material->F0 + (one - hit.material->F0) * powf(1.0f - cosa, 5);
			outRadiance = outRadiance + trace(Ray(hit.position + hit.normal * EPSILON, reflectedDir), depth + 1) * Fresnel;
		}

		return outRadiance;
	}

public:
	void build() {

		// Create camera
		vec3 eye = vec3{ -1.8f, 0.0f, 0.0f };
		vec3 vup = vec3{ 0.0f, 0.0f, 1.0f };
		vec3 lookat = vec3{ 0.0f, 0.0f, 0.0f };
		float fov = 70.0f * PI_F / 180.0f;
		camera.set(eye, lookat, vup, fov);

		// Create lights
		ambientLight = vec3{ 0.8f, 1.0f, 1.0f };												// Skylight
		lights.push_back(new Light(vec3{ 10.0f, 0.0f, 3.0f }, vec3{ 0.6f, 0.6f, 0.6f }));		// Sunlight

		// Create materials
		Material* GOLD = new ReflectiveMaterial(vec3{ 0.17f, 0.35f, 1.5f }, vec3{ 3.1f, 2.7f, 1.9f });
		Material* SILVER = new ReflectiveMaterial(vec3{ 0.14f, 0.16f, 0.13f }, vec3{ 4.1f, 2.3f, 3.1f });
		Material* PURPLE = new RoughMaterial(vec3{ 0.4f, 0.2f, 0.4f });
		Material* BLUE = new RoughMaterial(vec3{ 0.2f, 0.2f, 0.6f });
		Material* ORANGE = new RoughMaterial(vec3{ 0.35f, 0.175f, 0.0f });

		// Create objects
		objects.push_back(new Hyperboloid(vec3{ 0.0f, 0.0f, 0.95f }, vec3{ 0.625f, 0.625f, 1.0f }, SILVER, vec3{ 1.0f, 0.95f, 2.5f }));			// Sun tunnel (objects[0])
		objects.push_back(new Ellipsoid(vec3{ 0.0f, 0.0f, 0.0f }, vec3{ 2.0f, 2.0f, 1.0f }, ORANGE, vec3{ 1.0f, -1.0f, 0.95f }));				// Room


		// Reflective objects
		objects.push_back(new Paraboloid(vec3{ 0.9f, -0.1f, 0.0f }, vec3{ 0.5f, 0.5f, 0.4f }, GOLD, vec3{ 1.0f, -1.0f, 1.0f }));				// Gold paraboloid
		objects.push_back(new Hyperboloid(vec3{ 0.2f, 0.6f, -0.5f }, vec3{ 0.1f, 0.1f, 0.3f }, SILVER, vec3{ 1.0f, -1.0f, 0.0f }));									// Silver hyperboloid

		// Rough objects
		objects.push_back(new Ellipsoid(vec3{ 0.0f, 0.2f, -0.7f }, vec3{ 0.15f, 0.15f, 0.3f }, PURPLE));													// Purple (rough) ellipsoid
		objects.push_back(new Ellipsoid(vec3{ 0.2f, -0.25f, -0.6f }, vec3{ 0.05f, 0.075f, 0.05f }, BLUE));												// Blue (rough) ellipsoid
		objects.push_back(new Ellipsoid(vec3{ 0.2f, -0.25f, -0.725f }, vec3{ 0.075f, 0.15f, 0.075f }, BLUE));												// Blue (rough) ellipsoid
		objects.push_back(new Ellipsoid(vec3{ 0.2f, -0.25f, -0.9f }, vec3{ 0.1f, 0.2f, 0.1f }, BLUE));											// Blue (rough) ellipsoid
		objects.push_back(new Ellipsoid(vec3{ 0.0f, 0.2f, -0.3f }, vec3{ 0.1f, 0.1f, 0.1f }, PURPLE));														// Purple (rough) sphere

		// Generate sample points on hyperboloid surface
		for (int i = 0; i < 10; i++) {
			for (int j = 0; j < 10; j++) {
				float xCoord = -0.625f + i * 0.125f;
				float yCoord = -0.625f + j * 0.125f;
				if (xCoord * xCoord + yCoord * yCoord < 0.625f * 0.625f) lightSourceSamples.push_back(vec3{ xCoord, yCoord, 0.95f });
			}
		}
	}

	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
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
