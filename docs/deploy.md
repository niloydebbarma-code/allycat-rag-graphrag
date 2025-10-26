# Deploying Allycat

We made Allycat easy to use, customize and deploy.

This guide will show you how to package AllyCat so it's ready to deploy.

**Note:** For modern Docker-based deployments with GraphRAG support, see [Docker Deployment Guide](docker-deployment-guide.md) which covers Cloud, Hybrid, and Local deployment modes.

**Table of Contents**



## 1 -  Setup Your Runtime

You can run AllyCat either in:

- [native python](running-natively.md)
- or [docker environment](running-in-docker.md)

## 2 - Run AllyCat

[Run allycat](running-allycat.md)

This step will download and prepare content.  And we will run query.


## 3 - Package it for deployment

- if running in docker env: [package docker](package-docker.md)
- if running on native python: [package native](package-native.md)


## 4 - Publish the Docker Image

Now that we have packaged our customized AllyCat, it's time to deploy it.

```bash
docker  login 

# tag the image to match dockerhub account
# Replace "USER" with your dockerhub username (e.g. 'sujee')
docker image tag  allycat    USER/allycat
# e.g.
docker image tag  allycat-aialliance-deploy    sujee/allycat-aialliance-deploy

# push it
docker  push   USER/allycat-aialliance-deploy
# e.g.
docker  push   sujee/allycat-aialliance-deploy
```

## 5 -   Running the DockerHub image

Once published, anyone can pull and run the docker image

```bash
docker run -it --rm -p 8080:8080   sujee/allycat-aialliance-deploy    deploy
```

Go to URL:  http://localhost:8080

## 6 - Deploy the application to Google Cloud

See [deploy-google-cloud-run.md](deploy-google-cloud-run.md)