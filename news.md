# Allycat News

## 2025-08-31: Expanded Team

In addition to Sujee and Dave, we have begun working with Chirag, Nikhil and the team at Open Governance.

## 2025-07-14: Big Update 

Lot of cool updates:

**Robust web crawler** ([#31](https://github.com/The-AI-Alliance/allycat/issues/31))

Completely redid web crawler.  Now it 
- is more robust and handle scenarios that made previous crawler fail.
- can handle multiple file types (not just text/html) correctly
- Handle anchor tags (`a.html#news`) in HTML files correctly 
- pauses (customizable) between requests so to not hammer the webserver.
  
**using [LiteLLM](https://docs.litellm.ai/docs/) for LLM inference** ([#34](https://github.com/The-AI-Alliance/allycat/issues/34))

This allows us to seamlessly access LLMs running locally (using [ollama](https://ollama.com/)) or calling inference providers like Nebius, Replicate ..etc.

Also singinficantly simplified LLM configuration.

**Expanded support for many file types (pdf, docx)** ([#37](https://github.com/The-AI-Alliance/allycat/issues/37))

Before we just handled HTML files. Now we can download and process other popular file types - like PDF, DOCX ..etc.  We use [Docling](https://github.com/docling-project/docling) for processing files.


**Added [uv](https://docs.astral.sh/uv/) package manager support** ([#26](https://github.com/The-AI-Alliance/allycat/issues/26))

UV will be the preferred package manager going forward.  We will still maintain `requirements.txt` to support other package managers.


**Better config management**([#19](https://github.com/The-AI-Alliance/allycat/issues/19))

Lot of user configuration can be set using `.env` file.  This simplifies config management and allows for easier and faster experimentation without changing code.


**Documentation update**

Various doc updates.

**Huge thanks to all the contributors**

- [Steven Pousty](https://github.com/thesteve0)  ([linkedin](https://www.linkedin.com/in/thesteve0/))
- [Santosh Borse](https://github.com/santoshborse)
