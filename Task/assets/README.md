# AppWorld ReAct prompt

`appworld_react_generator_prompt.txt` is the unchanged public prompt from
[ACE AppWorld](https://github.com/ace-agent/ace-appworld/blob/928e86877d34cd10eaba159606386f93a1765090/experiments/prompts/appworld_react_generator_prompt.txt),
at commit `928e86877d34cd10eaba159606386f93a1765090`. It is distributed under
the upstream [Apache-2.0 license](LICENSE.appworld-prompt).

The demonstration account details and passwords are synthetic examples from
that public prompt. They are not runtime credentials. The prompt is preserved
byte for byte to retain the original experiment's demonstrations and instructions.

The adapter renders the supervisor and task instruction from the installed
AppWorld environment and supplies the selected memory method's retrieved context
as `playbook`. Set `APPWORLD_REACT_PROMPT_PATH` to use an explicit alternative.
The protected AppWorld task data and databases are downloaded separately and are
not redistributed here.
