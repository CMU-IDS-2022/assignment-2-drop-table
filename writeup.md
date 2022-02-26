# Project name

![A screenshot of your application. Could be a GIF.](screenshot.png)

TODO: Update screenshot

TODO: Short abstract describing the main goals and how you achieved them.

## Project Goals

TODO: **A clear description of the goals of your project.** Describe the question that you are enabling a user to answer. The question should be compelling and the solution should be focused on helping users achieve their goals. 

Through our application, we wanted to enable users to answer the question of how well certain countries are doing in their HIV response and preparedness. We wanted users to explore how well their own country or other countries and regions are doing in providing therapy coverage for HIV in response to growing HIV prevalence in the population or certain demographics in the population. For example, a user from Colombia could select the desired indiciator that they wish to explore and see its desired relationship with antiretroviral therapy coverage. From analyzing this relationship, the hope is to see that even though as incidences of HIV rises, the percentage of people living with HIV who are recieving HIV therapy treatments also rises. Additionally, to further exploration, we wanted to help users to explore growth and other changes in other related indicators such as current health expidenture and contraceptive use. From the interaction with our charts and utilizing the different indicators, we hope that users will be able to gauge how well their country and other countries are doing in response to HIV and be able to target areas for improvement.

## Design

TODO: **A rationale for your design decisions.** How did you choose your particular visual encodings and interaction techniques? What alternatives did you consider and how did you arrive at your ultimate choices?

A crucial part of our design was to filter and reduce the amount of data and features that were in the dataset we were using. Many of the features that were in the dataset were irrelevant and not very crucial to building a performance metric of HIV preparendess in countries. This is why we opted to use a single select dropdown menu so that users can select the relevant features they wanted to see at a time. We felt that the effort to fit too many features in our charts would also translate as increased difficulty in the cognitive understanding of our charts. However, we still wanted users to compare performance across multiple countries and over multiple years which is why a multiselect was used for countries to select the desired countries to analyze as well as a slider bar to show changes over the years. The suggestion to use the slider was brought up by Venkat who brought up the chart created by Hans Rosling in the Gapminder demo. We felt that since we were also given "timed" data that changes depending on what you select, it would also be interesting to apply that same principle to our charts. We initially tried using a bar chart or line chart for this interaction,however we found that using a scatter plot was more intuitive.

## Development

TODO: **An overview of your development process.** Describe how the work was split among the team members. Include a commentary on the development process, including answers to the following questions: Roughly how much time did you spend developing your application (in people-hours)? What aspects took the most time?

## Success Story

TODO:  **A success story of your project.** Describe an insight or discovery you gain with your application that relates to the goals of your project.
