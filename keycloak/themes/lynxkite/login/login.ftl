<#import "template.ftl" as layout>
<@layout.registrationLayout displayMessage=!messagesPerField.existsError('username','password') displayInfo=realm.password && realm.registrationAllowed; section>
    <#if section = "form">
        <h2 class="login-title">Sign in</h2>
        <#if realm.password>
            <form id="kc-form-login" onsubmit="login.disabled = true; return true;" action="${url.loginAction}" method="post">
                <div class="input-group">
                    <label for="username" class="input-label">
                        <#if !realm.loginWithEmailAllowed>${msg("username")}<#elseif !realm.registrationEmailAsUsername>${msg("usernameOrEmail")}<#else>${msg("email")}</#if>
                    </label>
                    <input tabindex="1" id="username" name="username" value="${(login.username!'')}" type="text" autofocus autocomplete="off"
                           class="login-input <#if messagesPerField.existsError('username','password')>input-error</#if>"
                           placeholder="Enter your username" />
                    <#if messagesPerField.existsError('username','password')>
                        <span class="input-error-message">${kcSanitize(messagesPerField.getFirstError('username','password'))?no_esc}</span>
                    </#if>
                </div>

                <div class="input-group">
                    <label for="password" class="input-label">Password</label>
                    <input tabindex="2" id="password" name="password" type="password" autocomplete="off"
                           class="login-input" placeholder="Enter your password" />
                </div>

                <div class="login-options">
                    <#if realm.rememberMe && !usernameEditDisabled??>
                        <label class="remember-me">
                            <input tabindex="3" id="rememberMe" name="rememberMe" type="checkbox"
                                   <#if login.rememberMe??>checked</#if>>
                            <span>Remember me</span>
                        </label>
                    </#if>
                    <#if realm.resetPasswordAllowed>
                        <a tabindex="5" href="${url.loginResetCredentialsUrl}" class="forgot-link">${msg("doForgotPassword")}</a>
                    </#if>
                </div>

                <input type="hidden" id="id-hidden-input" name="credentialId" <#if auth.selectedCredential?has_content>value="${auth.selectedCredential}"</#if>/>
                <button tabindex="4" name="login" id="kc-login" type="submit" class="login-button">
                    Sign In
                </button>
            </form>
        </#if>
    <#elseif section = "info">
        <#if realm.password && realm.registrationAllowed>
            <div class="register-link">
                ${msg("noAccount")} <a tabindex="6" href="${url.registrationUrl}">${msg("doRegister")}</a>
            </div>
        </#if>
    </#if>
</@layout.registrationLayout>
