[toc]

# MLOps - Made with ml

# Product

## 1 - Objective

- Identify the objective should be the *first* step in solving any problem.
- A useful tip: Try to identify the problem from the user's POV.

For example, if we study the following app.

This app has some articles and each article has tags which the users use for searching.

Our assigned problem is to : *improve search and discover-ability* as users have complained that they aren't able to discover stuff.

*PITFALL*: It is easy to **prematurely** jump to *tech jargon* such as better search algorithm / infrastructure / interface.

even though these are valid options, this might not resolve that **user's** issues. you need to focus more on the **user** and ask questions like

- What exactly are the complaints?
- Is it an issue of **content presence** or **discover-ability**? (may be incorrectly tagged or not tagged at all)
- Is there a specific way management wants to improve search?
- What past data do we have to work with? are the issues flagged?

****

##  2 - Solution

After Identifying our objective, we can hypothesize solutions using a 3-step process. (not a technical solution yet as it might not be needed).

1. **Visualize**

   Think of ideas from scratch **without** factoring **constraints**

2. **Understand**

   Understand how the problem is currently solved(if at all) and the **how** and **why** things are done that way.

   - prevents re-inventing the wheel.
   - gives insight into processes and signals available.
   - opportunity to question everything.

3. **Design**

   Design the solution while **factoring in constraints**

   - **Automation vs Augmentation**
     - Do we want to remove the user from the problem (automate) or help the user make decisions (augment)?
       - Be wary of removing the user
       - transition from augment to automate as trust grows
   - **UX constraints**
     - What are the aspects of UX that we need to respect?
       - privacy, personalization & digital real estate
       - dictate the components of our solution
   - **Technical constraints**
     - Details around the team, data and systems that you need to account for?
       - Availability of data, time , performance, cost, interpretability and latency
       - dictate the **complexity** of our solutions
   - 