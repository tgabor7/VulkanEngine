#pragma once

#include "lve_camera.hpp"
#include "lve_pipeline.hpp"
#include "lve_device.hpp"
#include "lve_game_object.hpp"
#include "lve_renderer.hpp"

//std
#include <memory>
#include <vector>


namespace lve {
	
	

	class SimpleRenderSystem {
	public:

		SimpleRenderSystem(LveDevice &device, VkRenderPass renderPass, LveRenderer &renderer);
		~SimpleRenderSystem();

		SimpleRenderSystem(const SimpleRenderSystem&) = delete;
		SimpleRenderSystem& operator=(const SimpleRenderSystem&) = delete;

		void renderGameObjects(VkCommandBuffer commandBuffer,VkDescriptorSet descriptorSet, std::vector<LveGameObject> &gameObjects, const LveCamera& camera);


	private:
		void createPipelineLayout();
		void createPipeline(VkRenderPass renderPass);
		
		LveDevice &lveDevice;
		VkPipelineLayout pipelineLayout;

		VkBuffer indexBuffer;
		VkDeviceMemory indexBufferMemory;

		LveRenderer& renderer;
		std::unique_ptr<LvePipeline> lvePipeline;
	};
}