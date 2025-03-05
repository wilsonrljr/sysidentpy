function openTab(evt, tabName) {
  const tabcontent = document.getElementsByClassName("tabcontent");
  for (let i = 0; i < tabcontent.length; i++) {
    tabcontent[i].style.display = "none";
  }
  const tablinks = document.getElementsByClassName("tablinks");
  for (let i = 0; i < tablinks.length; i++) {
    tablinks[i].classList.remove("active");
  }
  const currentTab = document.getElementById(tabName);
  if (currentTab) {
    currentTab.style.display = "block";
  } else {
    console.warn(`Error: Not found tab ${tabName}`);
  }
  if (evt.currentTarget && evt.currentTarget.classList) {
    evt.currentTarget.classList.add("active");
  } else {
    console.warn("Error element.");
  }
}
window.openTab = openTab;

const defaultOpenElement = document.getElementById("defaultOpen");
if (defaultOpenElement) {
  defaultOpenElement.click();
}

function installCode() {
  var elemento = document.getElementById("install");
  var selecao = window.getSelection();
  var intervalo = document.createRange();
  intervalo.selectNodeContents(elemento);
  selecao.removeAllRanges();
  selecao.addRange(intervalo);
  document.execCommand("copy");
  selecao.removeAllRanges();
}
window.installCode = installCode;

document.addEventListener("DOMContentLoaded", function () {
  const logo = document.querySelector("a.md-header__button.md-logo");

  if (logo) {
    const newLogo = document.createElement("a");
    newLogo.href = "./";
    newLogo.className = "md-header__button md-logo";
    newLogo.setAttribute("aria-label", "SysIdentPy");
    newLogo.setAttribute("data-md-component", "logo");
    newLogo.setAttribute("title", "SysIdentPy");

    const img = document.createElement("img");
    img.src = "overrides/assets/img/logotype-sysidentpy.svg";
    img.alt = "Logotype SysIdentPy";

    newLogo.appendChild(img);
    logo.replaceWith(newLogo);
  }
});
