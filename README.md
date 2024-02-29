# Building and Running the Dockerfile

To build and run the Dockerfile in this repository, you will need to have Docker installed on your machine. If you haven't installed Docker yet, you can download it from the [official Docker website](https://www.docker.com/products/docker-desktop).

## Building the Docker Image

Navigate to the directory containing the Dockerfile and run the following command:

```bash
docker build -t your_image_name .
```

Replace `your_image_name` with the name you want to give to your Docker image.

## Running the Docker Image

After building the Docker image, you can run it using the following command. 
The command mounts docs and files directories to the docker container so you can save your embeddings to disk.

Ensure you have docs and files directories created and configured accordingly.

```bash
docker run --env-file=.env -v $(pwd)/docs:/app/docs -v $(pwd)/files:/app/files -p 8501:8501  <your image>
```

This command will start a Docker container from your image and map port 8501 of the container to port 8501 of your host machine. The `--env-file .env` option is used to pass environment variables from the `.env` file to the Docker container.

# Environment Variables

The `envfile` file contains several environment variables that are used to configure the application. Here's a brief description of each variable:

- `MODE`: This variable determines the mode of operation. Supported modes are "azure-openai", "groq" and "openai". If you are using Azure OpenAI installation, set this to "azure-openai". Otherwise, leave it as "openai". 

- `MODEL_NAME`: This is a comma-separated list of model names that you want to support. Users will be able to pick one while querying the data. This is not applicable to Azure OpenAI deployments.

- `OPENAI_API_KEY`: This is the API key for OpenAI or Azure OpenAI models.

- `AZURE_OPENAI_API_KEY`: This is the API key for Azure configurations.

- `AZURE_OPENAI_DEPLOYMENT_NAME`: This is the deployment name for Azure configurations.

- `AZURE_OPENAI_ENDPOINT`: This is the endpoint for Azure configurations.

- `AZURE_OPENAI_VERSION`: This is the version for Azure configurations.
- `GROQ_API_KEY` : This is the api key for Groq API

Remember to replace the placeholder values in the `envfile` file with your actual keys and configurations before running the Docker container.


Create a file named envfile and add the values for following:

```
# supported modes are "azure-openai" , "openai". If you are using Azure OpenAI installation, use "azure-openai" or else leave it as "openai"

MODE=openai

# comma separated model names that you want to support. Users will be able to pick one while querying the data.
# not applicable to azure openai depolyments
MODEL_NAME=gpt-4, gpt-3.5-turbo-0125

# configurations if you are using openai model directly
OPENAI_API_KEY=

# azure configurations
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_DEPLOYMENT_NAME=
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_VERSION=
```


# How to use the application

Go to [http://localhost:8501](http://localhost:8501) in your browser and start using the application.