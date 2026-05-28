<#import "template.ftl" as layout>
<@layout.registrationLayout displayMessage=!messagesPerField.existsError('username','password'); section>

    <#if section = "form">

        <div class="login-card">
            <h2 class="login-title">${msg("doLogIn")}</h2>

            <#if messagesPerField.existsError('username','password')>
                <div class="alert alert-error">
                    ${kcSanitize(messagesPerField.getFirstError('username','password'))?no_esc}
                </div>
            </#if>

            <form id="kc-form-login" action="${url.loginAction}" method="post">

                <div class="input-group">
                    <label class="input-label" for="username">
                        <#if !realm.loginWithEmailAllowed>
                            ${msg("username")}
                        <#elseif !realm.registrationEmailAsUsername>
                            ${msg("usernameOrEmail")}
                        <#else>
                            ${msg("email")}
                        </#if>
                    </label>
                    <input
                        id="username"
                        name="username"
                        type="text"
                        class="login-input"
                        value="${(login.username!'')}"
                        autocomplete="username"
                        autofocus
                        placeholder="<#if !realm.loginWithEmailAllowed>${msg("username")}<#elseif !realm.registrationEmailAsUsername>${msg("usernameOrEmail")}<#else>${msg("email")}</#if>"
                    />
                </div>

                <div class="input-group">
                    <label class="input-label" for="password">${msg("password")}</label>
                    <input
                        id="password"
                        name="password"
                        type="password"
                        class="login-input"
                        autocomplete="current-password"
                        placeholder="${msg("password")}"
                    />
                </div>

                <div class="login-options">
                    <#if realm.rememberMe>
                        <label class="remember-me">
                            <input type="checkbox" name="rememberMe" <#if login.rememberMe??>checked</#if>>
                            ${msg("rememberMe")}
                        </label>
                    </#if>
                    <#if realm.resetPasswordAllowed>
                        <a href="${url.loginResetCredentialsUrl}" class="forgot-link">${msg("doForgotPassword")}</a>
                    </#if>
                </div>

                <input type="hidden" name="credentialId"
                    <#if auth.selectedCredential?has_content>value="${auth.selectedCredential}"</#if> />

                <button type="submit" class="login-button">${msg("doLogIn")}</button>

            </form>

            <#if realm.password && realm.registrationAllowed && !registrationDisabled??>
                <div class="register-link">
                    ${msg("noAccount")} <a href="${url.registrationUrl}">${msg("doRegister")}</a>
                </div>
            </#if>
        </div>

    </#if>

</@layout.registrationLayout>
