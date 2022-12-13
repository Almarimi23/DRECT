const axios = require("axios");
const moment = require("moment");
const ObjectsToCsv = require("objects-to-csv");
const parse = require("csv-parse/lib/sync");
const fs = require("fs");
// const strip = require("strip-comments");
const log = console.log;

const dataLocation = "scraped_data challenges full.csv";
const allCsvData = fs.readFileSync(dataLocation).toString();

//calling the npm package and save to records
const records = parse(allCsvData, {
  columns: true,
  skip_empty_lines: true,
});

//map the output from csv-parse to the column
const allCriteria = records.map(
  (data) =>
    `${data["id"]} ${data["winner"]} ${data["projectId"]} ${data["subTrack"]} ${data["runner up"]}`
);

// log(allCriteria);

const getReg = async (id) => {
  const { data } = await axios.get(
    `https://api.topcoder.com/v5/resources?challengeId=${id}`,
    {
      headers: {
        "User-Agent": "PostmanRuntime/7.28.4",
        Accept: "*/*",
        "Accept-Encoding": "gzip, deflate, br",
        Connection: "keep-alive",
      },
    }
  );

  const registrants = Object.keys(data)
    .map((val) => data[val])
    .map((item) => item["memberHandle"])
    .join(", ");

  const appliedDates = Object.keys(data)
    .map((val) => data[val])
    .map((item) => item["created"])
    .join(", ");

  return [appliedDates, registrants];
};

const getStartAndEndDate = async (id) => {
  const { data } = await axios.get(
    `https://api.topcoder.com/v5/challenges/${id}`,
    {
      headers: {
        "User-Agent": "PostmanRuntime/7.28.4",
        Accept: "*/*",
        "Accept-Encoding": "gzip, deflate, br",
        Connection: "keep-alive",
      },
    }
  );

  const { created: challengecreatedDate, submissionEndDate } = data;
  // duration = `${moment(endDate).diff(startDate, "days")} day`;
  return [challengecreatedDate, submissionEndDate];
};

const getSkills = async (handle) => {
  const { data } = await axios.get(
    `https://api.topcoder.com/v3/members/${handle}/skills`,
    {
      headers: {
        "User-Agent": "PostmanRuntime/7.28.4",
        Accept: "*/*",
        "Accept-Encoding": "gzip, deflate, br",
        Connection: "keep-alive",
      },
    }
  );

  const dataObject = data?.["result"]?.["content"]?.["skills"];
  const dataArray = Object.keys(dataObject)
    .map((val) => dataObject[val])
    .map((item) => item["tagName"])
    .join(", ");
  return dataArray;
};

const getSats = async (handle) => {
  const { data } = await axios.get(
    `https://api.topcoder.com/v5/members/${handle}/stats`,
    {
      headers: {
        "User-Agent": "PostmanRuntime/7.28.4",
        Accept: "*/*",
        "Accept-Encoding": "gzip, deflate, br",
        Connection: "keep-alive",
      },
    }
  );

  const infoObj = data?.[0]?.["DEVELOP"]?.["subTracks"];

  const avgInfo = (obj1, atr) => {
    const mappedInfo1 = obj1.map((item) => item.submissions[atr]);
    const totalInfo1 = mappedInfo1.reduce((total, next) => total + next, 0);
    return ((totalInfo1 / mappedInfo1.length) * 100).toFixed(2) || 0;
  };

  const winPercent = avgInfo(infoObj, "winPercent");

  const submissionRate = avgInfo(infoObj, "submissionRate");

  const reviewSuccessRate = avgInfo(infoObj, "reviewSuccessRate");

  const screeningSuccessRate = avgInfo(infoObj, "screeningSuccessRate");

  const avgReliability = (obj1, atr) => {
    const mappedInfo1 = obj1.map((item) => item["rank"][atr]);
    let total = 0;

    mappedInfo1.forEach((item) => {
      if (item) {
        total = total + item;
      }
    });
    return ((total / mappedInfo1.length) * 100).toFixed(2);
  };

  const reliability = avgReliability(infoObj, "reliability");

  const { challenges, wins } = data[0];

  return [
    challenges,
    wins,
    winPercent,
    reviewSuccessRate,
    submissionRate,
    reliability,
    screeningSuccessRate,
  ];
};

const getAge = async (handle) => {
  const { data } = await axios.get(
    `https://api.topcoder.com/v5/members/${handle}`,
    {
      headers: {
        "User-Agent": "PostmanRuntime/7.28.4",
        Accept: "*/*",
        "Accept-Encoding": "gzip, deflate, br",
        Connection: "keep-alive",
      },
    }
  );

  const { createdAt, maxRating } = data;
  const registerdDate = new Date(createdAt).toLocaleString();
  const rating = maxRating?.["rating"];

  return [registerdDate, rating];
};
const getCompletedCha = async (handle) => {
  const { data } = await axios.get(
    `https://api.topcoder.com/v4/members/${handle}/challenges/?filter=status%3DCompleted&limit=50&offset=50`,
    {
      headers: {
        "User-Agent": "PostmanRuntime/7.28.4",
        Accept: "*/*",
        "Accept-Encoding": "gzip, deflate, br",
        Connection: "keep-alive",
      },
    }
  );

  const { result } = data;
  const completedCha = result?.["metadata"]?.["totalCount"];

  return completedCha;
};
const getActiveCha = async (handle) => {
  const { data } = await axios.get(
    `https://api.topcoder.com/v4/members/${handle}/challenges/?filter=status%3DActive&limit=50&offset=50`,
    {
      headers: {
        "User-Agent": "PostmanRuntime/7.28.4",
        Accept: "*/*",
        "Accept-Encoding": "gzip, deflate, br",
        Connection: "keep-alive",
      },
    }
  );

  const { result } = data;
  const ActiveCha = result?.["metadata"]?.["allChallengesCount"];

  return ActiveCha;
};

const finalRes = [];
// empty list use to keep track of exiting user
// const listOfHandles = [];
(async () => {
  try {
    let i = 0;
    for (let idAndWinner of allCriteria) {
      try {
        // if (i > 2) break;
        const id = idAndWinner.split(" ")[0];
        const winner = idAndWinner.split(" ")[1];
        const projectId = idAndWinner.split(" ")[2];
        const type = idAndWinner.split(" ")[3];
        const runnerUp = idAndWinner.split(" ")[4];

        const [appliedDate, registrants] = await getReg(id);

        const allRegistrants = registrants
          .split(",")
          .map((item) => item.trim());
        const allAppliedDates = appliedDate
          .split(",")
          .map((item) => item.trim());

        for (let j = 0; j < allAppliedDates.length; j++) {
          log(j);
          log(id, allRegistrants[j]);
          try {
            // this lines of code update the filter
            //   if (listOfHandles.indexOf(allRegistrants[j]) > -1) {
            //     log("skippng duplicate");
            //     continue;
            //   }
            //   listOfHandles.push(allRegistrants[j]);
            // log(listOfHandles)

            const skills = await getSkills(allRegistrants[j]);
            const [
              challenges,
              wins,
              winPercent,
              reviewSuccessRate,
              submissionRate,
              reliability,
              screeningSuccessRate,
            ] = await getSats(allRegistrants[j]);

            const [registerdDate, rating] = await getAge(allRegistrants[j]);

            const [challengecreatedDate, submissionEndDate] =
              await getStartAndEndDate(id);
            // log(submissionEndDate, "\n", id);
            const activechallenges = await getActiveCha(allRegistrants[j]);
            const completedchallenges = await getCompletedCha(
              allRegistrants[j]
            );

            finalRes.push({
              challengeId: id,
              handle: allRegistrants[j],
              projectId,
              skills,
              total_number_skills: skills?.split(",").length,
              totalChallengsJoined: challenges,
              memberSince: registerdDate.split(" GMT")[0],
              "member life":
                moment
                  .duration(
                    moment()
                      .startOf("day")
                      .diff(
                        moment(registerdDate.split(" GMT")[0], "DD-MM-YYYY")
                      )
                  )
                  .asDays() ||
                moment
                  .duration(
                    moment()
                      .startOf("day")
                      .diff(
                        moment(registerdDate.split(" GMT")[0], "MM-DD-YYYY")
                      )
                  )
                  .asDays(),
              "topcoder rating": rating,
              registrationDate: allAppliedDates[j],
              // number of days after last registrationdate
              Duration: moment
                .duration(
                  moment(submissionEndDate, "YYYY-MM-DD").diff(
                    moment(challengecreatedDate, "YYYY-MM-DD")
                  )
                )
                .asDays(),
              totalWins: wins,
              winner: winner === allRegistrants[j] ? "yes" : "no",
              "Ativity Type": type,
              "number of days after registration date": moment
                .duration(
                  moment(submissionEndDate, "YYYY-MM-DD").diff(
                    moment(allAppliedDates[j], "YYYY-MM-DD")
                  )
                )
                .asDays(),
              activechallenges,
              completedchallenges,
              "Win Percentage": winPercent || 0,
              "review Success Rate": reviewSuccessRate || 0,
              "submission Rate": submissionRate || 0,
              reliability: reliability || 0,
              "screening Success Rate": screeningSuccessRate || 0,
              "runner Up": runnerUp,
            });
          } catch (error) {
            log(error.message, "skipping due to error");
            continue;
          }
        }
        i++;
      } catch (error) {
        log(error.message, "skipping due to error");
        continue;
      }
    }

    // log(finalRes);
    log("saving ....");
    new ObjectsToCsv(finalRes).toDisk("./scraped_data member.csv", {
      allColumns: true,
    });
  } catch (error) {
    log("saving ....");
    new ObjectsToCsv(finalRes).toDisk("./scraped_data member.csv", {
      allColumns: true,
    });
    log(error.message);
  }
})();

// https://api.topcoder.com/v5/challenges/?legacyId=30215504

// https://api.topcoder.com/v5/members?sort=desc&perPage=100&page=1
// https://api.topcoder.com/v5/members/lqz/stats?fields=userId,handle,wins,groupId,challenges
// https://api.topcoder.com/v5/members/lqz/skills?fields=userId,handle,handleLower,skills,createdAt,updatedAt,createdBy,updatedBy   https://api.topcoder.com/v5/members/lqz/skills
