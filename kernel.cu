#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <conio.h>
#include <Python.h>
#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>

#define BLOCK_SIZE 32

// Device variables, which help to compute and update boundary conditions
__device__ float box_temperatures_sum, space_temperatures_sum;
__device__ int box_elements, space_elements;
__device__ float box_mean_temperature, space_mean_temperature;


// Set initial values of space and box temperatures
__global__ void space_init(int space_dim, int box_dim, float water_temperature,
                            float box_temperature, float *u0_space, float *u0_box)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if (i < space_dim && j < space_dim)
		u0_space[i * space_dim + j] = water_temperature;

	if (i == 0 || j == 0 || i == box_dim - 1 || j == box_dim - 1)
		u0_box[i * box_dim + j] = 0.99 * box_temperature;
	else if(i < box_dim && j < box_dim)
		u0_box[i * box_dim + j] = box_temperature;

}


// Put box in space
__global__ void box_init(int space_dim, int box_dim, float *u0_space, float *u0_box)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if (i >= space_dim - box_dim && i < space_dim &&
		j >= (space_dim - box_dim) / 2 && j < (space_dim + box_dim) / 2)
	{
		int x = i - (space_dim - box_dim);
		int y = j - (space_dim - box_dim) / 2;
		u0_space[i * space_dim + j] = u0_box[x * box_dim + y];
	}
}


// Make single step in heat equation using finite difference method
__global__ void heat_equation(int space_dim, int box_dim, float dw, float dk, float dx2,
							float dy2, float dt, float *u0_space, float *u_space)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	// Pick proper diffusivity depending on space point (water or steel box)
	float d = dw;

	if (i > space_dim - box_dim && i < space_dim - 1 &&
		j > (space_dim - box_dim) / 2 && j < (space_dim + box_dim) / 2)
		d = dk;
	else if(i >= 1 && i < space_dim - 1 && j >= 1 && j < space_dim - 1)
		d = dw;

	// Update all points in space except box boundaries
	if ((i == space_dim - box_dim && j >= (space_dim - box_dim) / 2 && j < (space_dim + box_dim) / 2) ||
		(i > space_dim - box_dim && i < space_dim - 1 &&
		(j == (space_dim - box_dim) / 2 || j == (space_dim + box_dim) / 2 - 1)))
		u_space[i * space_dim + j] = u0_space[i * space_dim + j];
	else if (i >= 1 && i < space_dim - 1 && j >= 1 && j < space_dim - 1)
	{
		float uxx = (u0_space[(i + 1) * space_dim + j] - 2 * u0_space[i * space_dim + j] +
			u0_space[(i - 1) * space_dim + j]) / dx2;

		float uyy = (u0_space[i * space_dim + j + 1] - 2 * u0_space[i * space_dim + j] +
			u0_space[i * space_dim + j - 1]) / dy2;

		u_space[i * space_dim + j] = u0_space[i * space_dim + j] + dt * d * (uxx + uyy);
	}
}


// Rewrite data from buffer to main array
__global__ void rewrite_data(int space_dim, float *u0_space, float *u_space)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if(i >= 1 && i < space_dim - 1 && j >= 1 && j < space_dim - 1)
		u0_space[i * space_dim + j] = u_space[i * space_dim + j];
}


// Set device variables to zero, before summing temperatures in new iteration
__global__ void device_variables_to_zero()
{
	box_temperatures_sum = 0.0;
	space_temperatures_sum = 0.0;
	box_elements = 0;
	space_elements = 0;
}


// Sum space and box temperatures
__global__ void Temperature_Add(int space_dim, int box_dim, float *u0_space)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	// Use atomic operations to sum temperatures
	if (i > space_dim - box_dim && i < space_dim - 1 &&
		(j > (space_dim - box_dim) / 2 && j < (space_dim + box_dim) / 2 - 1))
	{
		atomicAdd(&box_temperatures_sum, u0_space[i * space_dim + j]);
		atomicAdd(&box_elements, 1);
	}
	else if (i >= 1 && i < space_dim - 1 && j >= 1 && j < space_dim - 1)
	{
		atomicAdd(&space_temperatures_sum, u0_space[i * space_dim + j]);
		atomicAdd(&space_elements, 1);
	}
}


// Calculate mean value of temperature of box and space 
__global__ void mean_temperature(float box_temperature, bool *end, bool *save, bool *save2)
{
	// Calculate mean value of temperatures
	box_mean_temperature = box_temperatures_sum / box_elements;
	space_mean_temperature = space_temperatures_sum / space_elements;

	// Decide whether save current state to csv file
	if (box_mean_temperature <= 0.9 * box_temperature)
		*save = true;
	if (box_mean_temperature <= 0.8 * box_temperature)
		*save2 = true;
	if (box_mean_temperature <= 0.7 * box_temperature)
		*end = true;
}


// Update boundary conditions
__global__ void update_boundary_conditions(int space_dim, int box_dim, float dk, float dt, float *u0_space)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if (((i == space_dim - box_dim) && j >= (space_dim - box_dim) / 2 && j < (space_dim + box_dim) / 2) ||
		(i > space_dim - box_dim && i < space_dim - 1 &&
		(j == (space_dim - box_dim) / 2 || j == (space_dim + box_dim) / 2 - 1)))
	{
		if (u0_space[i * space_dim + j] <= 274.0)
			u0_space[i * space_dim + j] = 274.0;
		else
			u0_space[i * space_dim + j] -= dk * dt * (box_mean_temperature - space_mean_temperature);
	}
}


// Save data to csv file
void save_to_file(FILE *csv_file, int space_dim, float * u0_space)
{
	for (int i = 0; i < space_dim; i++) 
		for(int j = 0; j < space_dim; j++)
			fprintf(csv_file, "%f,", u0_space[i * space_dim + j]);
}


// Execute python script with visualisation
void visualisation()
{
	char filename[] = "visualisation.py";
	FILE* python_file;

	int argc = 1;
	wchar_t* argv[1];
	argv[0] = L"visualisation.py";

	Py_Initialize();

	Py_SetProgramName(argv[0]);
	PySys_SetArgv(argc, argv);
	python_file = _Py_fopen(filename, "r");
	PyRun_SimpleFile(python_file, filename);

	Py_Finalize();
}


int main()
{
	// Start time measurement
	float time = 0.0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	// Init experiment conditions
	float water_temperature = 300.0;
	float box_temperature = 550.0;
	int box_dim = 12;
	int space_dim = 32;

	bool end = false;
	bool save = false;
	bool save2 = false;
	size_t space_size = space_dim * space_dim * sizeof(float);
	size_t box_size = box_dim * box_dim * sizeof(float);
	size_t bool_size = sizeof(bool);

	// Allocate memory for space and box
	float *u0_box, *u0_space, *u_space;
	u0_box = (float*)malloc(box_size);
	u0_space = (float*)malloc(space_size);
	u_space = (float*)malloc(space_size);

	// Allocate memory on device
	bool *dev_flag, *dev_save, *dev_save2;
	float *dev_u0_box, *dev_u0_space, *dev_u_space;
	cudaMalloc((void**)&dev_u0_space, space_size);
	cudaMalloc((void**)&dev_u_space, space_size);
	cudaMalloc((void**)&dev_u0_box, box_size);
	cudaMalloc((void**)&dev_flag, bool_size);
	cudaMalloc((void**)&dev_save, bool_size);
	cudaMalloc((void**)&dev_save2, bool_size);

	// Copy data from host to device
	cudaMemcpy(dev_u0_box, u0_box, box_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_u0_space, u0_space, space_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_flag, &end, bool_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_save, &save, bool_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_save2, &save2, bool_size, cudaMemcpyHostToDevice);
	
	// Init experiment space
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(space_dim / BLOCK_SIZE + 1, space_dim / BLOCK_SIZE + 1);
	space_init<<<dimGrid, dimBlock>>>(space_dim, box_dim, water_temperature, box_temperature, dev_u0_space, dev_u0_box);
	box_init<<<dimGrid, dimBlock>>>(space_dim, box_dim, dev_u0_space, dev_u0_box);
	cudaDeviceSynchronize();

	// Define variables for heat equation
	float dx = 0.1;
	float dy = 0.1;
	float dk = 58.0;
	float dw = 0.6;
	float dx2 = dx * dx;
	float dy2 = dy * dy;
	float dt = dx2 * dy2 / (2 * dk * (dx2 + dy2));

	int counter = 0;
	int saved = 0;

	FILE* csv_file;
	csv_file = fopen("visualisation_data.csv", "w");
	fprintf(csv_file, "%d\n", space_dim);

	while (!end && counter < 30000)
	{
		// Single step in heat diffusion
		heat_equation<<<dimGrid, dimBlock>>>(space_dim, box_dim, dw, dk, dx2, dy2, dt, dev_u0_space, dev_u_space);
		rewrite_data<<<dimGrid, dimBlock>>>(space_dim, dev_u0_space, dev_u_space);

		device_variables_to_zero<<<1, 1>>>();
		Temperature_Add<<<dimGrid, dimBlock>>>(space_dim, box_dim, dev_u0_space);
		mean_temperature<<<1, 1>>>(box_temperature, dev_flag, dev_save, dev_save2);
		update_boundary_conditions<<<dimGrid, dimBlock>>>(space_dim, box_dim, dk, dt, dev_u0_space);
		cudaDeviceSynchronize();

		cudaMemcpy(u0_space, dev_u0_space, space_size, cudaMemcpyDeviceToHost);
		cudaMemcpy(&end, dev_flag, bool_size, cudaMemcpyDeviceToHost);
		cudaMemcpy(&save, dev_save, bool_size, cudaMemcpyDeviceToHost);
		cudaMemcpy(&save2, dev_save2, bool_size, cudaMemcpyDeviceToHost);

		if (counter == 0 || (save && saved == 1) || (save2 && saved == 2))
		{
			save_to_file(csv_file, space_dim, u0_space);
			saved++;
		}

		counter++;
	}
	
	save_to_file(csv_file, space_dim, u0_space);
	fclose(csv_file);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	printf("Iteration count: %d\n", counter);
	printf("Execution time: %f s\n", time / 1000);

	// Free memory
	free(u0_box);
	free(u0_space);
	free(u_space);
	cudaFree(dev_flag);
	cudaFree(dev_save);
	cudaFree(dev_save2);
	cudaFree(dev_u0_box);
	cudaFree(dev_u0_space);
	cudaFree(dev_u_space);

	// Call visualisation script
	visualisation();

    return 0;
}
