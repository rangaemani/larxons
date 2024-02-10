# Larxons

Larxons is a neural network project inspired by [this post](https://pwy.io/posts/learning-to-fly-pt1/). It's written in Rust and relies on cargo, npm, webpack, and WASM.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Before you can run this project, you need to have the following tools installed on your machine:

- Rust and cargo: These are necessary to compile and run the Rust code. You can download them from [here](https://www.rust-lang.org/tools/install).
- Node.js and npm: These are necessary to run the web interface. You can download them from [here](https://nodejs.org/en/download/).
- wasm-pack: This is necessary to compile Rust to WebAssembly. You can download it from [here](https://rustwasm.github.io/wasm-pack/installer/).

Please ensure that you have these installed before proceeding with the "Installing" section.

### Installing

Clone the repository to your local machine:

```bash
git clone https://github.com/rangaemani/larxons.git
```

```bash
cd larxons/libs/simulation-wasm
```

```
wasm-pack build --release
```

```bash
cd ../../www
```

```bash
npm install
```

```bash
npm start
```
This will start a development server, and you can view the project in your browser at http://localhost:8080.

### Sample
![Image](/images/brave_A5EfPhxgjX.png)

