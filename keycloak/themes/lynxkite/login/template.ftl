<#macro registrationLayout bodyClass="" displayInfo=false displayMessage=true displayRequiredFields=false>
<!DOCTYPE html>
<html class="${properties.kcHtmlClass!}">
<head>
    <meta charset="utf-8">
    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />
    <meta name="robots" content="noindex, nofollow">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <#if properties.meta?has_content>
        <#list properties.meta?split(' ') as meta>
            <meta name="${meta?split('==')[0]}" content="${meta?split('==')[1]}"/>
        </#list>
    </#if>
    <title>${msg("loginTitle",(realm.displayName!''))}</title>
    <link rel="icon" href="${url.resourcesPath}/img/logo.png" />
    <#if properties.stylesCommon?has_content>
        <#list properties.stylesCommon?split(' ') as style>
            <link href="${url.resourcesCommonPath}/${style}" rel="stylesheet" />
        </#list>
    </#if>
    <#if properties.styles?has_content>
        <#list properties.styles?split(' ') as style>
            <link href="${url.resourcesPath}/${style}" rel="stylesheet" />
        </#list>
    </#if>
</head>
<body class="login-pf">
    <div class="login-wrapper">
        <div class="login-brand">
            <div class="logo-container">
                <img src="${url.resourcesPath}/img/logo.png" alt="LynxKite" class="logo-main" />
                <img src="${url.resourcesPath}/img/logo-sparky.jpg" alt="" class="logo-sparky" />
            </div>
            <div class="tagline">The Complete Graph Data Science Platform</div>
        </div>

        <div class="login-card">
            <#if displayMessage && message?has_content && (message.type != 'warning' || !isAppInitiatedAction??)>
                <div class="alert alert-${message.type}">
                    <span class="message-text">${kcSanitize(message.summary)?no_esc}</span>
                </div>
            </#if>

            <#nested "form">

            <#if displayInfo>
                <div class="login-info">
                    <#nested "info">
                </div>
            </#if>
        </div>

        <div class="login-footer">
            <span>Powered by LynxKite</span>
        </div>
    </div>
</body>
</html>
</#macro>
