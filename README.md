# MIB Cars Analysis

The goal of the sales process is to identify and communicate with potential leads, and explore opportunities to convert them into customers. For this project, the database contains records of every sales opportunity with information about the client, product, and process status which focuses on 3 key stages:-

Identified/Qualifying
Qualified/Validating
Validated/Gaining Agreement

This dataset consists of 78,025 rows and 19 columns (9 numerical, 10 categorical). The columns include the following:-

Amount - Estimated total revenue of opportunities in USD. (Goal 1)
Result - Outcome of opportunity. (Goal 2)
Id - A uniquely generated number assigned to the opportunity.
Supplies - Category for each supplies group.
Supplies_Sub - Sub group of each supplies group.
Region - Name of the region.
Market - The opportunities' route to market.
Client_Revenue - Client size based on annual revenue.
Client_Employee - Client size by number of employees.
Client_Past - Revenue identified from this client in the past two years.
Competitor - An indicator if a competitor has been identified.
Size - Categorical grouping of the opportunity amount.
Elapsed_Days - The number of days between the change in sales stages. Each change resets the counter.
Stage_Change - The number of times an opportunity changes sales stage. Includes backward and forward changes.
Total_Days - Total days spent in Sales Stages from Identified/Validating to Gained Agreement/Closing.
Total_Siebel - Total days spent in Siebel Stages from Identified/Validating to Qualified/Gaining Agreement.
Ratio_Identify - Ratio of total days spent in the Identified/Validating stage over total days in sales process.
Ratio_Validate - Ratio of total days spent in the Validated/Qualifying stage over total days in sales process.
Ratio_Qualify - Ratio of total days spent in Qualified/Gaining Agreement stage over total days in sales process.
