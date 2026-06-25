configfile: "config/config.transport.yaml"

include: "rules/common.smk"
include: "rules/hidrogeologia.smk"
include: "rules/vallereal.smk"


rule all:
    input:
        rules.hidrogeologia.input,
        rules.vallereal_all.input,
