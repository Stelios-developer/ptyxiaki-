const form = document.getElementById("analysis-form");
const message = document.getElementById("message");
const results = document.getElementById("results");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  message.textContent = "Η ανάλυση βρίσκεται σε εξέλιξη...";
  results.classList.add("hidden");

  try {
    const response = await fetch("/api/analyze", { method: "POST", body: new FormData(form) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail);
    showResults(data);
    message.textContent = "";
  } catch (error) {
    message.textContent = error.message || "Δεν ήταν δυνατή η ανάλυση του αρχείου.";
  }
});

function showResults(data) {
  const first = data.summary[0];
  document.getElementById("rows").textContent = data.rows.toLocaleString("el-GR");
  document.getElementById("attacks").textContent = first.attack.toLocaleString("el-GR");
  document.getElementById("rate").textContent = `${first.attack_rate.toFixed(2)}%`;
  document.getElementById("summary").innerHTML = summaryTable(data.summary);
  document.getElementById("bars").innerHTML = bars(data.summary);
  document.getElementById("preview").innerHTML = previewTable(data.preview_columns, data.preview_rows);
  document.getElementById("download").href = data.download;
  results.classList.remove("hidden");
}

function summaryTable(rows) {
  const body = rows.map((row) => `<tr><td>${row.model}</td><td>${row.normal}</td><td>${row.attack}</td><td>${row.attack_rate}%</td></tr>`).join("");
  return `<table><thead><tr><th>Μοντέλο</th><th>Normal</th><th>Attack</th><th>Ποσοστό attack</th></tr></thead><tbody>${body}</tbody></table>`;
}

function bars(rows) {
  return rows.map((row) => {
    const total = row.normal + row.attack;
    const normal = (row.normal / total * 100).toFixed(1);
    const attack = (row.attack / total * 100).toFixed(1);
    return `<div class="bar-row"><strong>${row.model}</strong><div><div class="bar-track"><div class="bar" style="width:${normal}%"></div></div><div class="bar-track"><div class="bar attack" style="width:${attack}%"></div></div></div><span>${attack}% attack</span></div>`;
  }).join("");
}

function previewTable(columns, rows) {
  const header = columns.map((column) => `<th>${column}</th>`).join("");
  const body = rows.map((row) => `<tr>${columns.map((column) => `<td>${row[column] ?? ""}</td>`).join("")}</tr>`).join("");
  return `<thead><tr>${header}</tr></thead><tbody>${body}</tbody>`;
}
