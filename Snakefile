configfile: "config/config.transport.yaml"

include: "rules/common.smk"
include: "rules/hidrogeologia.smk"


rule all:
    input:
        rules.hidrogeologia.input,
