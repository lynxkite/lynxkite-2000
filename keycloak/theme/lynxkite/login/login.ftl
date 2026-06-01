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
                    <div class="password-wrapper">
                        <input
                            id="password"
                            name="password"
                            type="password"
                            class="login-input"
                            autocomplete="current-password"
                            placeholder="${msg("password")}"
                        />
                        <button type="button" class="password-toggle" onclick="var p=document.getElementById('password');var t=p.type==='password'?'text':'password';p.type=t;this.classList.toggle('visible',t==='text');" aria-label="Toggle password visibility">
                            <svg class="eye-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/>
                                <circle cx="12" cy="12" r="3"/>
                            </svg>
                            <svg class="eye-off-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94"/>
                                <path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19"/>
                                <path d="M14.12 14.12a3 3 0 1 1-4.24-4.24"/>
                                <line x1="1" y1="1" x2="23" y2="23"/>
                            </svg>
                        </button>
                    </div>
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
