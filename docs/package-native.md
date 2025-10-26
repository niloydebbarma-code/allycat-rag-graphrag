# Dockerizing AllyCat (Running Natively)

This guide for packaging up Allycat when running it in 'native python dev' mode. If you are running AllyCat using docker, see [this guide](package-docker.md) instead.

**Note:** For modern Docker deployments with GraphRAG support and automatic pipeline execution, see [Docker Deployment Guide](docker-deployment-guide.md).

The packaged container will contain the following:

- Downloaded and cleaned web content
- Vector database that has indexed content (and optionally Neo4j graph database for GraphRAG)
- All runtime code and libraries
- Optionally, ollama model

## 1 - Run AllyCat

Go through all steps in [running AllyCat](running-allycat.md).

This will ensure all the web content is downloaded, cleaned and indexed.  And all models are working.

Make sure the UI at [localhost:8080](http://localhost:8080/) is working.

All artifacts will be in `./workspace` directory.  You can inspect its contents as follows

```bash
tree  workspace/
```

```text
workspace/
├── crawled
├── llama_index_cache
├── processed
└── rag_website_milvus.db
...
```

## 2 - Build a Docker container


```bash
docker  build  -t allycat-deploy0  .
```


## 3 - Running the docker container

```bash
docker run -it --rm -p 8080:8080 allycat-deploy0   deploy
```

We are supplying `deploy` option.

As the container starts, ollama will startup and download models.

Wait for this step to complete.

## 4 - Verify Functionality of the Docker

Thsi docker should have all the code and data and models ready to go.

Go to [localhost:8080](http://localhost:8080/) and try the chat.


## 5 - Packaging Allycat for deployment

Now that we have verified the container, let's package it for deployment.

**Make sure the AllyCat docker container is running; do not exit - you would loose downloaded artifacts**

Run these commands on another terminal.

**5.1 - `docker ps`**

Run `docker ps` command to find out the id of running AllyCat

Here is the sample output.  And the container id is `123456`

```text
CONTAINER ID   IMAGE     COMMAND                 CREATED          STATUS          PORTS                                                    NAMES
123456   allycat   "./docker-startup.sh"   17 minutes ago   Up 17 minutes   0.0.0.0:8080->8080/tcp, [::]:8080->8080/tcp, 11434/tcp   peaceful_elbakyan
```

**5.2 - Take a snaphot of the container**

```bash
docker commit    <container_id>    allycat-website-deploy

# eg
docker commit    123456    allycat-aialliance-deploy

# you can also add a version
docker commit    123456    allycat-aialliance-deploy:v1
```

**5.3 - Verify the snapshot is successfull**

```bash
docker   images
```

Sample output

```text
REPOSITORY                              TAG                  IMAGE ID       CREATED             SIZE
allycat-aialliance-deploy                          v1                   5ecf407ac875   3 minutes ago       6.83GB

```

**5.4 - Remove the container**

You can exit the previous allycat container

## 6 - Testing the deploy snapshot

Start the newly snapshotted docker image.


```bash
docker run -it --rm -p 8080:8080   allycat-aialliance-deploy    deploy
```

- the container being launched is `allycat-aialliance-deploy` - this is container we just snapshotted
- `deploy` option launches the web ui
- `p 8080:8080` makes sure the webui is available on `localhost:8080`

Go to url: [localhost:8080/](http://localhost:8080/)

And see if the chat is working.

## 6 - Continue to deploy

Now we are ready to deploy.

Continue to [deploy guide](deploy.md).