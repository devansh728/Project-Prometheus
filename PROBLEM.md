# SentinEV

## Problem Statement
   
   ### About the bussiness
A leading automotive OEM and service network in India provides aftersales maintenance services to a large customer base across metros and tier-2 cities. The business aims to increase customer retention, reduce vehicle breakdowns, optimize service center utilization, and improve manufacturing quality by proactively predicting maintenance needs, autonomously scheduling service appointments, and feeding insights back to the manufacturing team.

To achieve this, the company plans to deploy a web-based Agentic AI system acting as a Master Agent orchestrating multiple Worker AI agents to handle end-to-end predictive maintenance, customer engagement, service scheduling, and manufacturing quality improvement—using real-time vehicle data, historical maintenance logs, and CAPA/RCA records.

### Problem Statement

#### Business Problem: 
The company wants to improve vehicle uptime, enhance customer experience, and drive product quality improvements by:

Proactively predicting mechanical failures before they occur.
Autonomously scheduling service appointments to minimize unplanned downtime.
Leveraging RCA/CAPA insights from maintenance and manufacturing logs to improve design and reduce recurring defects.

#### Technical Solution: 

The company plans to deploy a web-based Agentic AI system acting as a Master Agent orchestrating multiple Worker AI agents to handle end-to-end predictive maintenance, customer engagement, service scheduling, and manufacturing quality improvement—using real-time vehicle data, historical maintenance logs, and CAPA/RCA records.

### Goal:
Design an Agentic AI solution where a Master Agent orchestrates multiple Worker AI agents to autonomously:

1) Continuously analyze real-time vehicle sensor data and historical maintenance logs using vehicle telematics
2) Predict upcoming mechanical issues using advanced diagnostics and failure prediction models.
3) Proactively contact vehicle owners with personalized maintenance recommendations primarily via voice-based agents, with mobile app notifications as a secondary channel. .
4) Forecast general service demand from maintenance history and vehicle usage patterns to optimize service center workloads and appointment planning.
5) Manage appointment scheduling by coordinating service center availability and customer preferences.
6) Track service progress until completion and follow-up for customer feedback.
7) Perform RCA/CAPA-driven analysis by cross-referencing predicted failures with historical maintenance and manufacturing defect records to suggest preventive actions, best-practice solutions, and feed insights back to manufacturing teams for quality improvement.
8) Ensure security and compliance by implementing UEBA (User and Entity Behaviour Analytics) for Agentic AI to monitor autonomous agent interactions, detect anomalies, and prevent unauthorized actions during orchestration. (Refer TIPS at the bottom for UEBA Understanding )

#### Must to have in Project:
1) Continuous vehicle monitoring and predictive failure detection.
2) Forecasting general service demand and autonomous scheduling based on vehicle usage and maintenance patterns.
3) Persuasive customer engagement via voice agent.
4) RCA/CAPA-based insights generation and feedback to manufacturing for quality improvement.
5) UEBA in action – detecting abnormal agent behavior or preventing unauthorized access.

#### Agentic AI Roles

A) Master Agent (Main Orchestrator)
1) Monitors vehicle health data streams and overall conversational flow.
2) Coordinates Worker Agents in diagnosis, customer outreach, scheduling, and feedback collection and insight feeding to manufacturing team
3) Ensures all agent interactions comply with security policies through UEBA-based anomaly detection
4) Initiates and ends customer interactions.

B) Worker Agents
1) **Data Analysis Agent**: Continuously analyzes streaming vehicle telematics and sensor data plus maintenance history to detect early warning signs and forecast likely maintenance needs or service demand
2) **Diagnosis Agent**: Runs predictive models to assess probable component failures and assigns priority levels.
3) **Customer Engagement Agent**: Initiates personalized conversations with vehicle owners via chatbot to explain predicted issues and recommend service.
4) **Scheduling Agent**: Checks service center capacity, proposes appointment slots, and confirms bookings with customers.
5) **Feedback Agent**: Follows up post-service to capture customer satisfaction and update vehicle maintenance records.
6) **Manufacturing Quality Insights Module**: Automatically generates actionable insights for the manufacturing team by analyzing predicted failures and historical CAPA/RCA data to improve product design and reduce recurring defects.

#### Data and System Assumptions:

1) **Synthetic Vehicle Data**: Data for 10 example vehicles including sensor readings, usage patterns, maintenance history, and diagnostic trouble codes.
2) **Telematics API**: Mock real-time sensor data feed.
3) **Maintenance Records Server**: Dummy database of historical repairs and service visits (can leverage open-source automotive datasets from Kaggle, UCI Repository, HuggingFace, etc.).
4) **Service Center Scheduler**: Mock API to retrieve available appointment slots and confirm bookings.
5) **Customer Interaction Layer**: Simulated voice-based virtual agent as the primary interface for owner communication, supplemented by app notifications for reminders and confirmations
6) **Security Layer**: UEBA integrated to monitor Master and Worker Agents for anomalous or malicious behaviours (e.g., unauthorized API calls or unexpected workflow changes).
7) **Forecasting & RCA Data**: Historical maintenance, manufacturing CAPA, and RCA records.

#### Evaluation Criteria

1) Technical design (40%)
    - Use of an Agentic AI framework (such as LangGraph, CrewAI, AutoGen) to orchestrate Master and Worker agents autonomously.
    - Incorporation of UEBA security measures for agent monitoring and anomaly detection.
2) Realism of data and workflow (25%)
    - Quality of synthetic telematics data, realistic failure prediction models, and simulated scheduling APIs.
3) Conversation flow (25%)
    - Natural, persuasive chatbot interaction explaining issues, answering queries, and closing service appointments.
4) Demo quality (10%)
    - Live demo or video walkthrough showing continuous vehicle monitoring, autonomous failure detection, customer engagement, RCA/CAPA insights, and scheduling from start to finish.
    - Demonstrate UEBA in action — for example, detecting and alerting on abnormal agent behaviour or blocking unauthorized API access.

#### Bonus Tips

1) **Emphasize persuasive and human-like chatbot/Voice agent conversations explaining the vehicle's condition and convincing owners to book services.**
2) **Showcase how the Master Agent coordinates real-time data analysis, predictive modeling, service demand forecasting, customer engagement, and appointment scheduling seamlessly, while integrating manufacturing feedback loops.**
3) **Demonstrate edge cases like declined appointments, urgent failure alerts, or multi-vehicle fleet scheduling, and recurring defects. Show how RCA/CAPA analysis informs better decision-making in these cases.**
4) **What is UEBA - UEBA uses advanced analytics and machine learning to establish behavioural baselines for users and entities (including AI agents) and detect anomalies that indicate potential threats or unauthorized activities.**
    - Example of UEBA: “In the predictive maintenance system, UEBA can monitor the Master Agent and Worker Agents for unusual activities—for instance, if the Scheduling Agent suddenly tries to access vehicle telematics data (which it normally doesn’t need), UEBA will flag this as anomalous behaviour and trigger an alert.”
    - Show how predicted failures and RCA/CAPA patterns are fed back to manufacturing teams to improve product design, reduce recurring defects, and close the loop between aftersales and production.
