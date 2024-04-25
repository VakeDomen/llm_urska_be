# URŠKA: Univerzitetne Rešitve: Študentski Komunikacijski Agent
# Server

## Overview

This backend socket server utilizes a Retrieval-Augmented Generation (RAG) model to assist university students by answering their inquiries in real-time, reducing the need for email communication with the student office.
System Requirements

- Rust Programming Environment (developed on: rustc 1.77.2 - 25ef9e3d8 2024-04-09)
- Adequate computational resources (CPU/GPU)
- Storage for model weights and data

Setup and Installation

- Clone the Repository:

```bash
git clone https://github.com/VakeDomen/llm_urska_be.git
cd student-assistant-server
```
- Install Dependencies:


```bash
cargo build
```

- Run the Server:


```bash
    cargo run
```

## Usage

Students can connect to the server through a compatible client application, submit their questions, and receive responses directly through the interface.
Contributing

## Contributions
We welcome contributions to improve the server. Please refer to the project's contributing guidelines for more information.
License

## License
This project is available under the MIT License. See the LICENSE file for more details.

## Contact

For further information, contact <domen.vake@famnit.upr.si>