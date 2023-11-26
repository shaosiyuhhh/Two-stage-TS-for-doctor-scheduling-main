# GA-VNS-for-Emergency-Department-doctor-scheduling
Hybrid two-stage and SA search algorithm, to solve the problem of doctor scheduling in emergency department.

### Problem Statement


- A day is divided into 24 hour-long periods on average. The defined starting point of a week is zero on Monday, assuming that no patients are in the system at this starting point.

- The arrival rate of patients can be different in different time periods, but the arrival rate of patients in a time period remains unchanged. The patient arrival rate per hour t is $\lambda_t$

- Service rate for each doctor is constant and all are $\mu$

- A doctor who is off duty and still has a patient on hand should finish the service of the patient before leaving his or her work.

- Emergency physicians can only commute at the beginning or end of each session. The patient queue is subject to the first come, first served rule (FCFS), assuming that there are no patients who leave without being served. The whole system can be thought of as a $𝑀_𝑡$ / $𝑀$ / $𝑝_𝑡$ queuing system (including $𝑝_𝑡$ needs to be determined by the doctor roaster).

- There are $N$ ($N = 15$) doctors in the emergency department

- Each doctor is allowed a maximum of two day shifts, or 1 night shift 24 hours a day starting at midnight

- Night shift: 0-7, during which doctors are not allowed to work. Night doctors are not allowed to work eight hours before the start of the night shift (after 17:00 the previous day) or on the day the night shift ends. A doctor must immediately rest for 24 hours straight after his night shift before he can be assigned a new job. 

- A doctor works no more than two night shifts a week.

- The maximum length of a doctor's shift is 8 hours and the minimum is 3 hours. Every doctor should take at least one full day off in a week, i.e., 24h (from 0:00 to 0:00 the next day) which can be fulfilled by the rest immediately after the night shift.

- At least one doctor is on duty at any one time serving patients

- Goal: Aim to minimize the weighted cost of the working time of all doctors and the waiting time of the patients in a week. Ask for each doctor's daily commute time.


### Run
Run `Two-stage-initial.py` to perform local search and gurobi solution adjustment get the initial solution of the scheduling and save the file in 'initial_result.xlsx'.

Run `TS_SA_search.py` to perform tabu-based Simulated Annealing adjustment to get the final solution of the scheduling and save the file in 'SA_search.xlsx'.

Run `Doctor_scheduling&line_fig.py` to draw the figure of the number of doctors and the waiting time during the planning horizon.

Run `TS_search_process_fig.py` to get the figure of the searching solution of the TS-SA algorithm.
