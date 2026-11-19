    Rapport

    Omfang 

    Målet med prosjektet er å utvikle en applikasjon som kan analysere odds for PL-kamper fra en bookmaker. Applikasjonen vil da bruke data fra PL-kamper fra perioden 1993-2023 for å trene en modell. Modellen vil da etter treningen kunna lage sannsynligheter for de mulige resultatene til kampen. Deretter sammenligner applikasjonen disse sannsynlighetene med de sannsynlighetene som tilsvarer oddsene fra bookmaker. 
    Maskinlæring kan være nyttig i denne oppgaven siden vi ønsker å bruke masse data for å forutse disse fotballkampene. Vi kan lage prediksjoner basert på data, men det er ikke alltid så lett å se sammenhengen mellom dataene og vite hvor mye de bør vektes opp mot hverandre. Ved å trene maskinlæringsmodeller kan de lære seg sammenhengene mellom statistikkenen og resultatene. Siden de kan beregne sannynlighet for forskjellige utfall er maskinlæring veldig egnet for denne oppgaven.
    I praksis så vil applikasjonen gi et valg av hvilken kommende fotballkamp brukeren vil analysere, så kan de sette inn odds fra en bookmaker som applikasjonen kan gi en vurdering av. Deretter henter applikasjonen relevante data fra en api som den kan gi til modellen sin. Som sagt vil da modellen beregne sannsynlighet for alle tre utfall og sammenligne de med sannsynligheten vi kan se i oddsene. Deretter kan applikasjonen si oss hvilken av resultatene som ser mest gunstige ut.
    Prosjektet er mest for de som er nysgjerrige. Modellen kan ikke gi noen garantier på gevinst og blir sannsynligvis slått av beregninger som blir gjort av større bookmakere som uansett gir seg selv greie marginer når det kommer til odds. 
    I dag blir alt maskinlæring brukt masse for å sette odds til kamper.[^1] Formålet til applikasjonen blir litt annerledes, men modellene vil nok ha stort sett like formål. 
    Det kan gjøres manuelt ved å se på data, kanskje lage et system av regler der man ser hvilken statistikker man bør vekte og regne seg frem til hvem som er favoritten. 

    Metrikker

    For å måle hvor bra modellen fungerer, brukes metrikkene accuracy, brier score og log loss. 
    Accuracy måler hvor ofte prediksjonene til modellen treffer. Dette gir oss et bra mål på om modellen klarer å forutsi kampresultater, men gir oss ikke noe mål på hvor gode sannsynlighetene den har regnet seg fram til er.
    Log loss gir oss et mål på hvor gode sannsynlighetene er. Den straffer modellen hardere dersom den tar feil på noe den har gitt veldig høy sannsynlighet, enn dersom den tar feil med et resultat den bare så vidt hadde som favoritt. 
    Brier score gjør i en viss grad det samme som log loss, men den gir oss ikke fullt så harde straffer på store feil, noe som kan gjøre modellen litt modigere.
    Formålet med å bruke både log loss og brier score som metrikk er å både sørge for at modellen ikke blir for selvsikker på enkelte prediksjoner, samtidig som vi heller ikke vil gjøre den feig og gir for jevne odds på grunn av straffer fra log loss.

    Brukertilfredstilthet vil være den viktigste business metrikken i dette prosjektet. Finner brukerene frem, opplever de applikasjonen som nyttig og interessant å bruke? Viss brukeren opplever dette og får bedre innsikt til oddsene på en fotballkamp kan man si at prosjektet har vært vellykket.

    Data

    Prosjektet bruker to datasett for to ulike formål. Det første er gamle kampdata fra kaggle. Det gir oss resultater fra 1993 frem til 2022 som vi kan bruke til å trene modeller. Det andre er data som vi henter fra en api. Api-en gir oss dataene som vi skal gi til den trente modellen. Dette blir da data fra de siste fem kampene til hjemmelaget og bortelaget.
    Fra datasettet får vi kampdata som sier oss hvordan et lag har prestert. Data som mål scoret, antall cornere, skudd, skudd på mål og selvsagt stilling/sluttresultat. 
    Det blir og utregnet en elo-rating til lagene som kan gi modellen mer forståelse for hvordan et lag har prestert. Et 2-1 tap mot et bra lag kan potenielt være "bedre" enn uavgjort mot et veldig svakt lag. Derfor kan det være nyttig å gi modellen et tall på styrken til lagene.
    Labelene til modellen er relativt rett frem å lage. Modellen skal klassifisere tre mulige utfall: hjemmeseier, uavgjort og borteseier. Vi får da labels direkte fra kampresultatene og vi trenger ikke å merke kamper manuelt. 
    Datasettet som brukes til trening inneholder data fra 11000 premier league kamper som ble spilt fra 1993/1994 sesongen til kamper som ble spilt i 2021/2022 sesongen. Antallet kamper vil nok være tilstrekkelig.
    
    Verken applikasjone eller modellen bruker noen form for sensitive personopplysninger. Det er forsåvidt viktig å si at det ikke gir noen garantier for å tjene penger eller slå oddsene fra bookmakeren. 

    Datasettet fra kaggle hadde en del manglende statistikker. Dette gjaldt først og fremst de tidligere sesongene, når man kommer til 2000-tallet er det ikke noe særlig hull i dataene. Det blir brukt medianverdi for å håndtere manglende data. 
    Dataene bearbeides slik at statistikken fra tidligere kamper brukes til å lage gjennomsnitt og andre egenskaper for lagene. 

    For å holde modellen relativt enkel brukes gjennomsnittet av de fem siste kampene for hjemmelag og bortelag. Det vil si ting som snitt av mål de fem siste kamper, i stedet for å bare gi modellen fem kamper i kronologisk rekkefølge. Det blir regnet ut seiersrate. Det vil antageligvis fortsatt gi modellen bra mål for å analysere hvordan et lag kan prestere.
    Elo-ratingen til lagene blir regnet ut med alle kampene frem til dit de er i sesongen der modellen skal lage prediksjon.
    Det blir benyttet standardscaler for å skalere data som blir gitt til logistic regression modellen.

    Modellering

    Det vi prøver å få modellen til å gjøre er en flerklassifiseringsoppgave.
    Det ble valgt to modeller som skulle trenes på dataene. Disse er logistic regression og random forest. 
    Logistic regression er bra egnet til å jobbe med klassifiseringsoppgave. Den kan trenes raskt og har relativt lav risiko for overfitting. Logistic regression går ut i fra at dataene er lineært sammenhengende. De fleste statistikkene vi gir modellen kan gås ut fra å være det. Høyere seiersrate og høyere antall scorede mål henger tett i lag med resultatet vi prøver å predikere.     
    Random forest kan gi et annet sett med fordeler. Den kan fange opp ikke-lineære sammenhenger bedre. Den kan se sammenhenger mellom flere ulike statistikker og er ikke fullt så sårbar for sjokkresultater. Den har lettere for overfitting enn logistic regression.
    Vi brukte en enkel modell som baseline. Den fungerte ved å gi en fast sannsynlighet basert på utfallsfordelingen i dataene og den gjettet alltid hjemmeseier. Baseline modellen ble tilpasset på treningsdataene (85% av settet) og så testet på testdataene (de gjenværende 15% av settet). Den oppnådde en accuracy på 43,1% og en log loss på 1,0745. Modellene som skulle trenes bør i allefall kunne slå dette for å si at de er gode nok.   
    Det ble logistic regression modellen som gav best resultat. Den oppnådde en accuracy på 53,35 %, en log loss på 0,9836 og en Brier score på 0,5827 på testsettet. Den slo dermed baseline-modellen på både accuracy (opp fra 43,10 %) og log loss (ned fra 1,0745).
    På valideringssettet var de to modellene ganske jevne, der Random Forest fikk 55,41 % accuracy og 0,9610 i log loss, mens Logistic Regression fikk 55,17 % accuracy og 0,9559 i log loss. Siden applikasjonen skulle sammenligne sannsynligheter med odds, valgte vi modellen med lavest log loss.

    Deployment

    Modellen er satt i drift som en streamlit applikasjon. Api-en henter et utvalg av kamper som er de kommende kampene i PL. Brukeren kan velge en kamp som de vil analysere. Dermed henter api-en dataene som skal brukest og gir de til modellen. Den regner ut de samme variablene som under trening: gjennomsnitt fra de siste fem kampene (mål, skudd, skudd på mål, cornere, frispark, kort, poeng og seiersrate), lagforskjeller og oppdatert Elo-rating basert på spilte kamper. Så regner applikasjonen ut hvilken sannsynligheter som tilsvarer oddsene som brukeren har satt inn. Disse sannsynlighetene blir sammenlignet med de som modellen gir i en tabell. Om en av modellens sannsynligheter er høyere enn bookmakerens sannsynlighet, slik at forventet verdi (Expected Value) blir over 5 %, gir applikasjonen beskjed om at dette ser ut som et gunstig bet. Expected value blir regnet ut ved å ta bookmakerens odds f.eks 1.6 og multiplisere det med modellens sannsynlighet f.eks 0,65 og trekke fra 1.

    kilder:

    [^1] https://www.lsports.eu/blog/machine-learning-in-odds-trading-where-it-works-and-fails