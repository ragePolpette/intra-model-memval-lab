# Target Objective - Intra Model Memory System

Aggiornato al: 2026-03-05
Scopo: fissare l'obiettivo finale per evitare drift e fraintendimenti.

## 1. Obiettivo finale (end-to-end)
Costruire un sistema completo di memoria intra-model con due componenti di storage e un ciclo cronico di adattamento pesi:

1. Componente `memory-save + evaluation` interna al runtime modello (quando supportato).
2. Componente `memory-save + evaluation` esposta via MCP per modelli open-weight che non supportano integrazione interna.
3. Architettura a due modelli:
   - Modello principale: produce output e salva memorie.
   - Modello analizzatore (stessi pesi iniziali del principale): legge memorie, le analizza e propone update limitato.
4. Aggiornamento controllato del principale: modifica circa `1.0% - 1.5%` dei pesi, con policy e gate di sicurezza.

## 2. Requisito chiave di parita'
Le due versioni della componente di salvataggio (`internal` e `MCP`) devono condividere lo stesso contratto logico:

- stessi campi semantici comuni (scoring, provenance, contesto, tracciabilita')
- stessa regola di self-evaluation/scoring
- stesse regole di mitigazione feedback loop
- stessi output per dataset export a parita' di input

Differenza ammessa e voluta:
- `internal`: payload primary numerico (`raw_numeric`)
- `MCP`: payload primary testuale (`text_view`) senza conversione numerica artificiale

In pratica resta allineata la semantica della memoria, cambia il tipo di payload disponibile.

## 3. Pipeline funzionale target
1. Writer model genera evento/memoria.
2. Storage layer valida + valuta + salva:
   - internal: salva numeric-native (+ text shadow opzionale)
   - MCP: salva text-native (primary)
3. Dataset builder applica filtri/policy anti-bias.
4. Analyzer model legge dataset memorie e calcola delta di adattamento.
5. Weight update engine applica update limitato (`<=1.5%` pesi target) al modello principale.
6. Gate di sicurezza verificano regressioni prima di consolidare.

## 4. Vincoli di sicurezza e qualita'
- Filtro novelty sempre applicato in dataset selection.
- Quota minima `is_external` garantita nel dataset.
- Distribuzione bucket (top/mid/low) controllata.
- Tracciabilita' completa: policy usata, versioni, manifest, metriche.
- Nessun update pesi senza valutazione gate pre/post update.

## 5. Cosa e' gia' coperto oggi
- Fondazione storage + self-eval + selection policy + export offline nel progetto `intra-model-memval-lab`.
- Base pronta per essere usata come reference implementation del contratto.

## 6. Cosa manca per arrivare al target completo
- Definire contratto unico ufficiale `internal <-> MCP`.
- Implementare orchestrazione del modello analizzatore cronico.
- Implementare motore update pesi limitato 1.0%-1.5%.
- Implementare gating automatizzato pre/post update e report.

## 7. Definition of Done (progetto completo)
Il progetto e' considerato completo quando:

1. Esistono entrambe le varianti storage (`internal` e `MCP`) con parita' funzionale verificata.
2. Il ciclo cronico analyzer -> update pesi del principale e' operativo.
3. L'update e' quantitativamente limitato (1.0%-1.5%) e auditabile.
4. I gate anti-feedback-loop e anti-regressione bloccano update non sicuri.
5. Esiste documentazione operativa di runbook e rollback.
