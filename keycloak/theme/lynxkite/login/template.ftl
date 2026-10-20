<#macro registrationLayout bodyClass="" displayInfo=false displayMessage=true displayRequiredFields=false>
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="robots" content="noindex, nofollow">
    <title>${msg("loginTitle",(realm.displayName!''))}</title>
    <link rel="stylesheet" href="${url.resourcesPath}/css/login.css" />
</head>
<body class="login-pf">

<div class="login-pf-page">
    <div class="container">

        <div class="login-brand-wrap">
            <img
                src="${url.resourcesPath}/img/logo.png"
                alt="LynxKite"
                style="width:240px;height:auto;filter:drop-shadow(0 0 20px rgba(57,188,243,0.3));"
                onerror="this.style.display='none';document.getElementById('lk-text-logo').style.display='block';"
            />
            <div id="lk-text-logo" style="display:none;">
                <div class="login-brand-logo">Lynx<span>Kite</span></div>
                <div class="login-brand-sub">2000:MM</div>
            </div>
            <div class="login-brand-tagline">Discover Drugs, Graphs and the World!</div>
        </div>

        <#nested "form">

        <#if displayInfo>
            <#nested "info">
        </#if>

    </div>
</div>

</body>
</html>
</#macro>
