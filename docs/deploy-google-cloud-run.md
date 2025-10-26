# Deploy to Google Cloud Run

**Note:** For modern Docker deployments with GraphRAG support, automatic pipeline execution, and memory optimization, see [Docker Deployment Guide](docker-deployment-guide.md) which includes Cloud Run deployment instructions.


## 1 - Setup Google Cloud CLI

Follow isntructions from [here](https://cloud.google.com/sdk/docs/install)

## 2 - Init Google Cloud Shell (One time setup)

```bash
# Log in to Google Cloud if running locally
gcloud auth login

## setup project id
gcloud   config   set project   YOUR_PROJECT_ID
# e.g.
# gcloud   config   set project  allycat-456220
```

## 3 - Enable APIs (One time Setup)

```bash
# Enable Cloud Run and Container Registry APIs
gcloud services enable run.googleapis.com   containerregistry.googleapis.com

# Configure Docker to use gcloud as a credential helper
gcloud auth configure-docker
```

## 4 - Build and Tag docker image

Go through [deployment checklist](deploy.md) and build the docker image that is ready to deploy.

Tag the image so it can be pushed to Google Container Registry

```bash
docker image tag  allycat-deploy    gcr.io/YOUR_PROJECT_ID/allycat-deploy

# e.g.
# docker image tag  allycat-deploy    gcr.io/allycat-456220/allycat-deploy
# docker image tag  allycat-aialliance-deploy    gcr.io/allycat-456220/allycat-aialliance-deploy
```

## 4 - Push image to Docker registry

push the image

```bash
docker push gcr.io/YOUR_PROJECT_ID/YOUR_APP_NAME

# e.g.
# docker push   gcr.io/allycat-456220/allycat-deploy
# docker push   gcr.io/allycat-456220/allycat-aialliance-deploy
```

## 5 - Deploy app to Cloud Run

```bash
gcloud   run   deploy   allycat \
  --image gcr.io/YOUR_PROJECT_ID/YOUR_APP_NAME \
  --platform managed \
  --region REGION \
  --port 8080 \
  --timeout 200 \
  --memory 8G \
  --cpu 8 \
  --allow-unauthenticated \
  --args="deploy"
```

Here is an example

```bash
gcloud   run   deploy   allycat \
  --image gcr.io/allycat-456220/allycat-deploy \
  --platform managed \
  --region us-central1 \
  --port 8080 \
  --timeout 200 \
  --memory 8G \
  --cpu 8 \
  --allow-unauthenticated \
  --args="deploy"
```

you can specify additional options like this

```bash
gcloud   run   deploy   allycat \
  --image gcr.io/YOUR_PROJECT_ID/YOUR_APP_NAME \
  --platform managed \
  --region us-central1 \
  --memory 8G \
  --cpu 8 \
  --min-instances=1 \
  --max-instances 5 \
  --concurrency 80 \
  --port 8080 \
  --set-env-vars "KEY1=VALUE1,KEY2=VALUE2" \
  --args="deploy"
```

## 6 - Check Running Cloud Run Instances

View and manage from [Google Cloud Run console](https://console.cloud.google.com/run)

Using CLI:

```bash
# list services
gcloud run services list

# delete a running service
gcloud run services delete allycat  --region us-central1 
```