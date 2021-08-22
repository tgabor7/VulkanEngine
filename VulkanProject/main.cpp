#include "first_app.hpp"
#include <iostream>
#include <stdexcept>
#include <vector>
#include <algorithm>

int main() {

	lve::FirstApp app{};

	try {
		app.run();
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << "\n";
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}