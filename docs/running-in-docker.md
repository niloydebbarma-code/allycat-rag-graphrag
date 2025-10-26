# Running Allycat Using Docker

This is the quickest way to try out AllyCat. No setup needed. Just need Docker runtime.

**Note:** This guide covers the basic Docker setup. For production deployments with GraphRAG support and multiple deployment modes (Cloud/Hybrid/Local), see [Docker Deployment Guide](docker-deployment-guide.md).

The docker container has:

- All code and python libraries installed
- Ollama installed for serving local models
- Python web UI installed (Flask and Chainlit interfaces available)

## Prerequisites:

[Docker](https://www.docker.com/) or compatible environment.

## Step-1: Pull the AllyCat Docker Image

```bash
docker pull sujee/allycat
```

## Start the AllyCat Docker

Let's start the docker in 'dev' mode

```bash
docker run -it --rm -p 8080:8080  -v allycat-vol1:/allycat/workspace  sujee/allycat
```

- `-p 8080:8080`: maps port 8080 to the web UI
- `-v allycat-vol1:/allycat/workspace` maps a volume into workspace directory.  This way all of our work (downloaded web content, models ..etc) would be saved.

After the container starts, you will be within the container in shell.

## Docker Container Layout

The working directory is `/allycat`

All downloaded artifacts such as website content, models will be under the `/allycat/workspace` directory.

## Running AllyCat

Continue to [running allycat](running-allycat.md)
