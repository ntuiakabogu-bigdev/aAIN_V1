#aAIN V1 - African Artificial Intelligence(V1)

## Pan-African Translation Infrastructure

## Overview
Open-source Pan-African speech-to-speech translation infrastrucure currently containing English and Nigerian langauges (Yoruba, Hausa, Igbo).

## First Breakthrough
This Infrastructure was used to fine-tune a baseline infrastructure using localized datasets. The results is a production-ready model for Nigerian and African dialects.
With the results below;
AIN v1.2 - Evaluation Report
============================================================
Test set size:       2711 pairs
Base model BLEU:     14.06
Fine-tuned BLEU:     26.86
Improvement:         +12.80

>**Note On Models:** To protect our propietary data, the fine-tuned models are not provided in this repository

## Core Features
**African-to-African Translation With English** Can translate, transcribe, perform Speech-to-Speech and real time translation for the following African Languages alongside English;
- Yoruba
- Hausa
- Igbo(Without Speech),
Therefore translating;
- Yoruba to English, Igbo(Text) and Hausa
- Hausa to English, Igbo(Text) and Yoruba
- Igbo(Text) to Yoruba, Hausa and English
**RESTful API** FastAPI-driven endpoints for real-time, speech-to-speech translation, easily integratable into web or mobile applications.

## Quick Start
**Clone the Repo:**
git clone https://github.com/ntuiakabogu-bigdev/aAIN_V1.git
cd aAIN_V1

## License 
This Infrastructure code is released under the **Apache License Version 2.0**
*Copyright (c) 2026 aAIN/ Akabogu Chinua-Ntui Onochie*