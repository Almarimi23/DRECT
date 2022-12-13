const axios = require("axios");
const moment = require("moment");
const ObjectsToCsv = require("objects-to-csv");
// const strip = require("strip-comments");
const log = console.log;

// let filterData
const strip = (str) => {
  if (str === null || str === "" || typeof str !== "string") return "";
  return str
    .toString()
    .replace(/(https?|ftp):\/\/[\.[a-zA-Z0-9\/\-]+/g, "")
    .replace(/<a\b[^>]*>(.*?)<\/a>/gi, "")
    .replace(/(<([^>]+)>)/gi, " ")
    .replace(/\n{1}/g, " ")
    .replace(/ {2,}/g, "")
    .replace(/&nbsp/g, "")
    .replace(/</g, "")
    .replace(/>/g, "");
};

const finalRes = [];
(async () => {
  try {
    const allLinks = [
      `https://api.topcoder.com/v5/challenges/?status=Completed&perPage=100&page=`,
      `https://api.topcoder.com/v5/challenges/?startDateEnd=2014-11-25&endDateStart=2010-07-23&status=Completed&perPage=100&page=`,
      `https://api.topcoder.com/v5/challenges/?startDateEnd=2010-07-22&status=Completed&perPage=100&page=`,
    ];
    for (let link of allLinks) {
      let i = 1;
      while (true) {
        log(i);

        const { data } = await axios.get(
          `${link}${i}&sortBy=startDate&sortOrder=desc&tracks[]=Dev&types[]=CH&types[]=F2F&types[]=TSK`,
          {
            headers: {
              "User-Agent": "PostmanRuntime/7.28.4",
              Accept: "*/*",
              "Accept-Encoding": "gzip, deflate, br",
              Connection: "keep-alive",
            },
          }
        );

        // if (i > 2) break;
        if (data.length < 1) break;

        data.forEach(
          async ({
            id,
            projectId,
            name,
            description,
            created,
            endDate,
            overview,
            tags,
            type,
            legacy,
            winners,
            numOfRegistrants,
          }) => {
            finalRes.push({
              id,
              projectId,
              name,
              created,
              endDate,
              description: strip(description.slice(0, 250)),
              duration: `${moment(endDate).diff(created, "days")} day`,
              totalPrizes: overview?.["totalPrizes"],
              Languages: tags?.join(", "),
              type,
              subTrack: legacy?.["subTrack"],
              winner: winners[0]?.["handle"],
              "runner up": winners[1]?.["handle"],
              "Number of winners": winners?.length,
              numOfRegistrants,
            });
          }
        );
        i++;
      }
    }

    const filterData = finalRes.filter(
      (filter) =>
        filter?.["subTrack"] === "FIRST_2_FINISH" ||
        filter?.["subTrack"] === "CODE" ||
        filter?.["subTrack"] === "ASSEMBLY_COMPETITION" ||
        filter?.["type"] === "First2Finish"
    );
    // log(filterData);
    log("saving ....");
    new ObjectsToCsv(filterData).toDisk("./scraped_data challenges full.csv", {
      allColumns: true,
    });
  } catch (error) {
    // log("saving ....");
    // new ObjectsToCsv(filterData).toDisk("./scraped_data.csv", {
    //   allColumns: true,
    // });
    log(error.message);
  }
})();
