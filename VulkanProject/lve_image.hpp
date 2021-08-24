#pragma once
#include "lve_device.hpp"


namespace lve {

	class LveImage {
		public:
			LveImage(LveDevice& device);
			~LveImage();

			static VkImageView createImageView(LveDevice& device, VkImage image, VkFormat format);

			void createImage(uint32_t width, uint32_t height, VkFormat format,
				VkImageTiling tiling, VkImageUsageFlags usage,
				VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory);
		private:
			void createTextureImage();
			VkCommandBuffer beginSingleTimeCommands();
			void endSingleTimeCommands(VkCommandBuffer commandBuffer);
			void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
			void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);
			void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);
			void createTextureImageView();
			void createTextureSampler();

			LveDevice& lveDevice;
			VkImageView textureImageView;
			VkImage textureImage;
			VkDeviceMemory textureImageMemory;

			VkSampler textureSampler;
		};
}