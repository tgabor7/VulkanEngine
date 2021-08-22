#pragma once
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <string>
#include <stdexcept>


using namespace std;

namespace lve {
	class LveWindow {
	public:
		LveWindow(const int &w,const int &h,const string &name);
		~LveWindow();

		LveWindow(const LveWindow&) = delete;
		LveWindow &operator=(const LveWindow&) = delete;

		bool shouldClose() { return glfwWindowShouldClose(window); }
		VkExtent2D getExtent() const { return { static_cast<uint32_t>(width), static_cast<uint32_t>(height) }; };
		bool wasWindowResized() { return frambufferResized; };
		void resetWindowResizedFlag() {
			frambufferResized = false;
		}
		GLFWwindow* getGLFWwindow() const {
			return window;
		}


		void createWindowSurface(VkInstance instance, VkSurfaceKHR* surface);
	private:
		static void framebufferResizeCallback(GLFWwindow *window, int width, int height);
		void initWindow();

		int width;
		int height;
		bool frambufferResized = false;

		std::string windowName;
		GLFWwindow *window;
	};
}