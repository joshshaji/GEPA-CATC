INSTRUCTION_PROMPT = "Your role is to serve as a policy model that given a task and a set of tools, selects some tools and "\
    "determines the execution plan of tools that can be executed in sequential or parallel to solve the task. "\
    "Your goal is to generate the tool plans that can optimize the task performance while minimizing the execution costs.\n"\

TOOL_PROMPT2_P1 = "Each tool has its own functionality. "\
    "Executing a tool will incur some execution costs. " \
    "Besides, the costs of each tool may vary based on the size of inputs. " \
    "The following are the embedding features of each tool:\n"
TOOL_PROMPT2_P2 = "\nThe corresponding cost features of each tool are as follows:\n"

TASK_PROMPT = "\nNext, you will receive information about the task and the attributes of the current inputs.\n"\
    "Task specifications: [Task Specification]\n"\
    "Task input attributes: [Task Input Attributes]\n"\
    "Now please generate a tool plan.\n"