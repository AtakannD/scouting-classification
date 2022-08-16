**Scoutium-Machine Learning and Scouting Classification**


1. Business Problem

Predicting which class (average, highlighted) players are according to the scores given to the characteristics 
of the football players watched by the Scouts.


2. Story of the Datasets

The data set consists of information from Scoutium, which includes the features and scores of the football 
players evaluated by the scouts according to the characteristics of the footballers observed in the matches.


3. Variables of Datasets

scoutium_attributes:
	task_response_id: The set of a scout's evaluations of all players on a team's roster in a match
	match_id: The id of the relevant match
	evaluator_id: The id of the evaluator(scout)
	player_id: The id of the relevant player
	position_id: The id of the position played by the relevant player in that match
	(1: Goalkeeper
	2: Stopper
	3: Right back
	4: Left-back
	5: Defensive midfielder
	6: Central midfielder
	7: Right wing
	8: Left wing
	9: Offensive midfielder
	10: Striker)
	analysis_id: Set of attribute evaluations of a scout for a player in a match attribute_id: The id of 
each attribute by which the players were evaluated
	attribute_value: The value (points) a scout gives to a player's attribute 

scoutium_potential_labels:
	ask_response_id: The set of a scout's evaluations of all players on a team's roster in a match
	match_id: The id of the relevant match
	evaluator_id: The id of the evaluator(scout)
	player_id: The id of the relevant player
	potential_label: Label that indicates the final decision of a scout regarding a player in a match.

Task 1: Preprocessing and Manipulating Data 

Task 2: Encoding and Scaling Data

Task 3: Creating Machine Learning Model




