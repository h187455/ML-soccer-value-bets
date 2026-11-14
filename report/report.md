# Premier League Value Bet Analyzer

Mathias Fåberg
13. november 2026

1. BESKRIVELSE AV PROBLEMET

### 1.1 Omfang og mål

Målet med prosjektet var å utvikle en applikasjon som benytter maskinlæring for å analysere fotballkamper og gi oss prediksjoner for hvordan de går. Applikasjonen skal estimere sannsynligheten for de tre mulige kampresultatene: hjemmeseier, uavgjort og borteseier. Deretter vil vi gi applikasjonen odds for de ulike resultatene fra en bookmaker, så vil applikasjonen si oss om den tenker oddsene er gunstige eller ikke.

I sportsbetting representerer odds bookmakerens vurdering av sannsynligheten for et bestemt utfall, justert for bookmakerens margin. Det kan derfor være interessant å undersøke om en maskinlæringsmodell kan identifisere situasjoner der dens estimerte sannsynlighet avviker fra sannsynligheten som ligger implisitt i bookmakerens odds.

Løsningen er utviklet som en Python-applikasjon med Streamlit som brukergrensesnitt. Brukeren velger en kommende Premier League-kamp fra en liste og legger inn bookmakerodds for hjemmeseier, uavgjort og borteseier. Applikasjonen henter deretter relevant kampdata fra et eksternt API, beregner egenskaper for de to lagene og sender disse til en ferdig trent maskinlæringsmodell. Resultatet presenteres som sannsynligheter og forventet verdi for hvert av de tre utfallene.

Maskinlæring er relevant for dette problemet fordi kampresultater påvirkes av mange faktorer samtidig. Eksempler er lagenes tidligere prestasjoner, målsnitt, skuddstatistikk, poengfangst og relative styrke. Det er krevende å vurdere alle disse faktorene manuelt på en konsistent måte. En maskinlæringsmodell kan lære statistiske sammenhenger fra historiske kamper og bruke disse til å estimere sannsynligheten for fremtidige resultater.

En alternativ løsning uten maskinlæring ville vært å utvikle et regelbasert system. Dette kunne for eksempel gitt et lag en høyere vurdering basert på flere seiere, flere scorede mål eller bedre Elo-rating. En slik løsning ville imidlertid krevd at sammenhengen mellom hver faktor og kampresultatet ble definert manuelt. Maskinlæring gir mulighet til å lære disse sammenhengene fra data i stedet.

Prosjektet er først og fremst en demonstrasjon av hvordan maskinlæring kan brukes til å analysere sportsdata og støtte beslutninger. Det er ikke et mål å garantere lønnsom betting. For å kunne vurdere om systemet faktisk gir økonomisk verdi, må det gjennomføres mer omfattende testing over tid.

### 1.2 Metrikker

Modellens ytelse vurderes ved hjelp av flere metrikker:

- **Accuracy:** Andelen kamper der modellen predikerer riktig kampresultat. Dette gir et enkelt mål på hvor ofte modellen velger riktig klasse.
- **Log loss:** Måler hvor godt modellens estimerte sannsynligheter samsvarer med de faktiske resultatene. Feilaktige prediksjoner med høy sikkerhet straffes spesielt hardt.
- **Brier score:** Måler avstanden mellom predikerte sannsynligheter og faktiske utfall. En lavere verdi indikerer bedre sannsynlighetsestimering.
- **Expected Value (EV):** Brukes i applikasjonen for å vurdere bookmakerodds opp mot modellens sannsynligheter.

For et utfall med sannsynlighet \(P\) og desimalodds \(O\) beregnes forventet verdi slik:

\[
EV = P \cdot O - 1
\]

En positiv forventet verdi betyr at modellens sannsynlighet gir en høyere beregnet verdi enn bookmakeroddsen tilsier. Dette forutsetter imidlertid at modellens sannsynligheter er tilstrekkelig presise og godt kalibrerte.

Accuracy alene er derfor ikke tilstrekkelig for å vurdere prosjektets mål. Siden applikasjonen skal sammenligne sannsynligheter med odds, er log loss og Brier score også viktige. En modell kan ha relativt god accuracy, men likevel gi dårlige sannsynlighetsestimater.

Et langsiktig mål er at modellen skal kunne identifisere kamper med positiv forventet verdi på en konsistent måte. Dette må undersøkes gjennom historisk backtesting, der modellens prediksjoner sammenlignes med faktiske bookmakerodds og kampresultater.

## 2. DATA

### 2.1 Datakilder og datatyper

Prosjektet benytter to hovedkilder til data.

Den første kilden er et historisk datasett med Premier League-kamper fra sesongene 1993/94 til 2021/22. Datasettet inneholder omtrent 11 113 kamper og inkluderer blant annet kampdato, hjemmelag, bortelag, målscore og kampresultat. For en stor del av kampene finnes det også statistikk som antall skudd, skudd på mål, cornere, frispark og kort.

Den andre kilden er SportAPI.ai, som brukes til å hente informasjon om nåværende og kommende kamper. API-et gir tilgang til laginformasjon, kampresultater og kampstatistikk. Dette gjør det mulig for applikasjonen å hente oppdaterte data uten at kampinformasjonen må registreres manuelt.

Dataene består hovedsakelig av strukturerte numeriske og kategoriske verdier. Eksempler på numeriske verdier er antall mål og skudd, mens lagnavn og kampresultater er kategoriske verdier.

### 2.2 Labels

Prosjektet benytter supervised learning. Modellen trenes på historiske kamper der det faktiske kampresultatet er kjent.

Labelen er kampresultatet, representert med tre klasser:

- **H:** Hjemmeseier
- **D:** Uavgjort
- **A:** Borteseier

Labelen bestemmes ut fra sluttresultatet. Dersom hjemmelaget scorer flere mål enn bortelaget, blir resultatet H. Dersom lagene scorer like mange mål, blir resultatet D. Dersom bortelaget scorer flest mål, blir resultatet A.

Dette gir tydelige og konsistente labels som kan utledes direkte fra sluttresultatet. Det er derfor ikke nødvendig å merke kampene manuelt.

### 2.3 Databehandling og feature engineering

Før dataene kan brukes til modelltrening, må de renses og transformeres. Dette inkluderer blant annet å standardisere kolonnenavn, konvertere datoer og numeriske verdier til riktige datatyper og håndtere manglende verdier.

Et sentralt steg er feature engineering. Målet er å omforme rå kampdata til egenskaper som beskriver lagenes prestasjoner før en aktuell kamp.

For hver kamp beregnes gjennomsnitt fra lagets fem foregående kamper. Dette inkluderer blant annet:

- Gjennomsnittlig antall scorede og innslupne mål.
- Gjennomsnittlig antall skudd og skudd på mål.
- Gjennomsnittlig antall cornere, frispark og kort.
- Gjennomsnittlig antall poeng.
- Seiersandel.
- Antall tidligere kamper.

Poeng beregnes etter det vanlige ligasystemet, der seier gir tre poeng, uavgjort gir ett poeng og tap gir null poeng.

Det beregnes også Elo-rating for å representere lagenes relative styrke. Alle lag starter med en rating på 1500. Ratingen oppdateres etter kampene basert på forventet og faktisk resultat. Hjemmelaget får i beregningen en fordel på 100 Elo-poeng. Oppdateringen bruker en K-faktor på 20, som bestemmer hvor mye ratingen endres etter en kamp.

I tillegg beregnes differansevariabler for egenskapene. Disse representerer forskjellen mellom hjemmelagets og bortelagets statistikk. Dette gjør det mulig for modellen å bruke den relative forskjellen mellom lagene som informasjon.

En viktig del av feature engineering er å unngå datalekkasje. Når egenskapene for en historisk kamp beregnes, skal kun informasjon fra tidligere kamper være tilgjengelig. Resultatet og statistikken fra selve kampen må ikke brukes til å predikere den samme kampen.

Manglende verdier håndteres ved hjelp av median-imputering i modellens pipeline. Dette er særlig relevant for API-dataene, der ikke alle statistikkategoriene er tilgjengelige for alle kamper. Dersom en statistikk mangler, behandles den som en manglende verdi i stedet for å bli satt til null.

### 2.4 Personvern og etiske hensyn

Prosjektet bruker kampstatistikk og informasjon om fotballag. Det behandles ikke personopplysninger om vanlige brukere, og løsningen har derfor begrensede personvernrisikoer.

Det finnes likevel etiske hensyn knyttet til bruksområdet. Betting innebærer økonomisk risiko, og modellens prediksjoner kan være feil. Applikasjonen må derfor presenteres som et analyseverktøy og ikke som en garanti for gevinst. Forventet verdi er et matematisk estimat basert på modellens sannsynligheter, og resultatene er avhengige av kvaliteten på både dataene og modellen.

## 3. MODELLERING

### 3.1 Valg av modeller

Prosjektet undersøkte tre ulike maskinlæringsmodeller:

1. Logistic Regression
2. Random Forest
3. Gradient Boosting

Logistic Regression er en klassifikasjonsmodell som kan brukes til å estimere sannsynligheter for flere klasser. Modellen fungerer som et godt utgangspunkt og er forholdsvis enkel å tolke.

Random Forest er en ensemblemodell som kombinerer flere beslutningstrær. Den kan fange opp ikke-lineære sammenhenger og samspill mellom egenskaper.

Gradient Boosting bygger også på beslutningstrær, men trener modellene sekvensielt slik at senere trær forsøker å rette opp feil fra tidligere trær.

For å sammenligne modellene ble datasettet delt kronologisk i tre deler:

- 70 % treningsdata.
- 15 % valideringsdata.
- 15 % testdata.

En kronologisk oppdeling ble valgt fordi fotballprediksjoner skal brukes på fremtidige kamper. En tilfeldig oppdeling kunne ført til at modellen ble evaluert på kamper som tidsmessig lå før kamper i treningsdatasettet.

Modellene ble først trent på treningsdataene og sammenlignet ved hjelp av valideringsdataene. Log loss ble brukt som hovedkriterium for modellvalg, fordi prosjektet er opptatt av sannsynlighetsestimering og ikke bare korrekt klassifisering. Den best presterende modellen ble deretter trent på nytt med både trenings- og valideringsdata, før den ble evaluert på det separate testsettet.

### 3.2 Baseline og evalueringsresultater

For å vurdere om maskinlæringsmodellen gir en forbedring, ble resultatene sammenlignet med en enkel baseline-modell. Baseline-modellen bruker en enkel strategi basert på klassefordelingen i treningsdataene, i stedet for å lære sammenhenger mellom kampens egenskaper.

Den siste registrerte evalueringen viste følgende resultater:

| Metrikk | Logistic Regression | Baseline |
|---|---:|---:|
| Accuracy | 53,31 % | 43,10 % |
| Log loss | 0,9836 | 1,0745 |
| Brier score | 0,5827 | Ikke registrert |

Logistic Regression oppnådde bedre accuracy og lavere log loss enn baseline-modellen. Dette tyder på at egenskapene som er konstruert fra kampdataene gir nyttig informasjon om kampresultatene.

Logistic Regression ble derfor valgt som modell for applikasjonen. Den lagres som en pipeline bestående av median-imputering, standardisering av numeriske egenskaper og Logistic Regression med opptil 2000 iterasjoner.

Resultatene viser likevel at det fortsatt er betydelig usikkerhet i prediksjonene. Fotballkamper påvirkes av mange forhold som ikke nødvendigvis finnes i datasettet. Selv en modell som presterer bedre enn baseline vil derfor ofte predikere feil.

### 3.3 Feilanalyse og begrensninger

En utfordring ved modellen er at de tre resultatklassene ikke er like enkle å predikere. Uavgjort er særlig vanskelig å identifisere. I en tidligere evaluering hadde modellen ingen korrekt klassifiserte uavgjortresultater i testsettet, selv om den fremdeles kunne tildele uavgjort en viss sannsynlighet.

Dette illustrerer forskjellen mellom klassifisering og sannsynlighetsestimering. Modellen velger én klasse som mest sannsynlig, men det betyr ikke at sannsynligheten for de andre klassene er null.

En annen begrensning er forskjellen mellom det historiske datasettet og dataene fra SportAPI.ai. Det historiske datasettet dekker mange sesonger, mens API-et gir tilgang til en mer begrenset mengde nyere kamper. Elo-ratingene beregnes ved å samle fullførte Premier League-kamper som API-et returnerer for lagene, fjerne duplikater og oppdatere ratingene kronologisk. Kamper på samme dato bruker ratingene fra før den aktuelle datoen.

Fordi API-historikken er begrenset, kan Elo-ratingene være mindre etablerte enn ratingene som ble beregnet fra det historiske treningsdatasettet. Det kan derfor oppstå en forskjell mellom Elo-verdiene modellen ble trent på og verdiene den mottar i applikasjonen.

I tillegg er enkelte statistikker, som cornere og røde kort, ikke alltid tilgjengelige fra API-et. Disse verdiene håndteres som manglende data og fylles inn av modellens imputeringssteg. Dette gjør at applikasjonen kan fungere selv om enkelte statistikker mangler, men det kan også redusere kvaliteten på prediksjonene.

Det er også viktig å skille mellom modellens prediksjonsevne og dens evne til å identifisere lønnsomme spill. En modell som presterer bedre enn baseline på historiske kampresultater, er ikke nødvendigvis lønnsom når den brukes sammen med bookmakerodds. Dette må undersøkes separat gjennom backtesting.

## 4. DEPLOYMENT

### 4.1 Applikasjon og bruk

Modellen er integrert i en applikasjon utviklet med Streamlit. Applikasjonen fungerer som et brukergrensesnitt mellom brukeren, SportAPI.ai og den ferdig trente maskinlæringsmodellen.

Når applikasjonen startes, lastes den lagrede modellen fra en Joblib-fil. Deretter hentes tilgjengelige lag og kommende kamper fra SportAPI.ai. Brukeren velger en kamp og legger inn bookmakerodds for de tre mulige utfallene.

Når brukeren starter analysen, henter systemet lagenes siste tilgjengelige kamper og tilhørende statistikk. Disse dataene brukes til å beregne egenskapene som modellen trenger. Egenskapene organiseres deretter i samme kolonnerekkefølge som modellen forventer.

Modellen returnerer sannsynligheter for hjemmeseier, uavgjort og borteseier. Disse sannsynlighetene kombineres med oddsene brukeren har oppgitt, og forventet verdi beregnes for hvert utfall.

Resultatene vises i en tabell med bookmakerodds, bookmakerens implisitte sannsynlighet, modellens sannsynlighet og forventet verdi. Applikasjonen fremhever også utfallet med høyest forventet verdi. Dersom den høyeste forventede verdien overstiger fem prosent, viser applikasjonen dette som et mulig spill. Dette er en valgt visningsgrense i applikasjonen, ikke en dokumentert garanti for lønnsomhet.

### 4.2 Teknologier og drift

Prosjektet er utviklet med Python. Streamlit brukes til brukergrensesnittet, Pandas til behandling av data, Requests til API-kommunikasjon og Scikit-learn til maskinlæringsmodellen. Den ferdig trente modellen lagres og lastes ved hjelp av Joblib.

SportAPI.ai brukes som ekstern datakilde. API-nøkkelen lagres separat i Streamlits secrets-konfigurasjon, slik at den ikke trenger å ligge direkte i kildekoden.

API-kommunikasjonen har også en enkel mekanisme for å forsøke forespørsler på nytt dersom det oppstår tidsavbrudd eller tilkoblingsfeil. Elo-ratingene beregnes fra tilgjengelige kamper og mellomlagres i én time for å unngå å gjenta hele beregningen ved hver oppdatering av applikasjonen.

Applikasjonen er avhengig av at SportAPI.ai er tilgjengelig og returnerer data i forventet format. Endringer i API-et, manglende statistikk eller begrensninger i datatilgangen kan derfor påvirke funksjonaliteten.

### 4.3 Vedlikehold og videreutvikling

For å holde systemet relevant bør både datagrunnlaget og modellens ytelse følges opp over tid. Nye kamper bør samles inn og brukes til å evaluere om modellens sannsynligheter fortsatt samsvarer med faktiske resultater.

Et viktig forbedringsområde er å få tilgang til en mer komplett historikk for Elo-beregningene. Det bør også undersøkes om modellen bør trenes på nytt med et Elo-oppsett som samsvarer bedre med dataene som brukes i applikasjonen.

Andre mulige forbedringer er å:

- Kalibrere modellens sannsynligheter bedre.
- Undersøke hvorfor uavgjort er vanskelig å predikere.
- Sammenligne flere modeller og ulike kombinasjoner av egenskaper.
- Gjennomføre backtesting mot historiske bookmakerodds.
- Undersøke om nye egenskaper, som skader og lagoppstillinger, kan forbedre prediksjonene.

En særlig viktig videreutvikling vil være å teste om positive forventede verdier faktisk gir positiv avkastning over et større antall kamper. Dette krever historiske odds, en tydelig innsatsstrategi og at man tar hensyn til bookmakerens margin.

## 5. REFERANSER

[1] SportAPI.ai. Dokumentasjon for API og fotballdata. https://sportapi.ai/docs

[2] Scikit-learn. Dokumentasjon for Logistic Regression og modellpipelines. https://scikit-learn.org/

[3] Streamlit. Dokumentasjon for utvikling av Streamlit-applikasjoner. https://docs.streamlit.io/

[4] [Oppgi navn og lenke til det historiske Premier League-datasettet som ble brukt.]

[5] [Oppgi eventuelle pensumressurser eller fagartikler om Elo-rating og evaluering av klassifikasjonsmodeller.]