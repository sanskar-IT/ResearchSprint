# ResearchSprint: Collaborative AI Innovation Agent 🚀
### Kaggle Agents Intensive Capstone Project

**Team:** Sanskar Malviya
**Track:** Enterprise Agents 
## 📖 Project Pitch
**Problem:** Innovation teams suffer from context switching—moving between research, brainstorming, and drafting reports breaks flow and introduces bias.  
**Solution:** ResearchSprint is a multi-agent system that simulates a full expert team (Planner, Scout, Critic, Scribe) to autonomously turn a one-line goal into a peer-reviewed, comprehensive technical report.

## 🏗️ Architecture
* **Orchestrator:** Central control loop managing state.
* **Knowledge Scout:** Uses DuckDuckGo to verify facts.
* **Ideation Agent:** Generates hypothesis.
* **Critique Agent:** Reviews content for logical fallacies.
* **Scribe Agent:** Formats final output.

## 🛠️ Tech Stack
* **Model:** Google Gemini 2.5 Flash
* **Framework:** Google ADK-style workflow with Gemini models
* **Environment:** Kaggle Notebooks

## 🚀 How to Run
1. Clone this repository.
2. Install dependencies: `pip install -q -U google-generativeai ddgs`
3. Add your Gemini API Key to secrets.
4. Run `google-capstone-project-submission.ipynb`.
5. Run tests: `python -m unittest discover -s tests -p 'test*.py' -v`.

## 📄 Example Output
(https://docs.google.com/document/d/1Xu-JQUZ-4uvUhZK0b1ufb6vhIG_87W-3/edit?usp=sharing&ouid=117318804700665677451&rtpof=true&sd=true)]

## Video Demonstration
https://youtu.be/J0jIUDwZS0k
