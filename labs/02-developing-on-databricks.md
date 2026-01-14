- [Developing on Databricks](#developing-on-databricks)
  - [Lecture 2 of MLOps with Databricks course](#lecture-2-of-mlops-with-databricks-course)
  - [Getting Started](#getting-started)

# Developing on Databricks

![opening](../media/mlops-loop-en.jpg)

## Lecture 2 of MLOps with Databricks course

Special thanks to Maria Vechtomova 🌟

Databricks recently introduced Free Edition, which opened the door for us to create a free hands-on course on MLOps with Databricks.

This article is part of the course series, where we walk through the tools, patterns, and best practices for building and deploying machine learning workflows on Databricks :

* Lecture 1: Introduction to MLOPs
* Lecture 2: Developing on Databricks
* Lecture 3: Getting started with MLflow
* Lecture 4: Log and register model with MLflow
* Lecture 5: Model serving architectures
* Lecture 6: Deploying model serving endpoint
* Lecture 7: Databricks Asset Bundles
* Lecture 8: CI/CD and deployment strategies
* Lecture 9: Intro to monitoring
* Lecture 10: Lakehouse monitoring

Most people using Databricks start by developing directly in a Databricks notebook, because it’s easy, fast, and convenient. But when it comes to MLOps, that convenience can quickly become a bottleneck. Notebooks make it difficult to write modular code, apply proper code quality standards, or run unit tests, all of which are essential for maintainable, production-grade ML systems.

Fortunately, there’s a better way. Databricks developer tools, such as VS Code extension, Databricks CLI, and Databricks Connect, allow you to develop locally using modern engineering workflows, while still running your pyspark code on Databricks.

In this lecture, we’ll show you how to use these tools to move development outside notebooks and adopt workflows that align better with MLOps practices.

## Getting Started

To follow along with the course, you’ll need to set up a few things. In the video, we demonstrate how to walk through these steps.

1. Get the Databricks free edition. The course uses Databricks free edition. Do not confuse it with Databricks free trial, it is not the same thing!

2. Fork the course repo: https://github.com/marvelousmlops/marvel-characters. Forking the repo will allow you to work on CI/CD pipeline later in the course. Clone the forked repo on your local machine.

3. Create catalogs and schemas. In the Databricks free edition workspace, create catalogs mlops_dev, mlops_acc, and mlops_prd.Under each catalog, create schema marvel_characters. These catalogs and schemas are required to run the code.

4. Install the CLI. Databricks has very good documentation on how to install it: https://docs.databricks.com/en/dev-tools/cli/install.html. From our experience, the homebrew option works great on MacOS. On Window — winget. Otherwise, you can always install from a source build.

5. Authenticate towards Databricks. Databricks CLI should be used to authenticate towards Databricks from your local machine.