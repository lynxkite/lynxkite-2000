<#import "template.ftl" as layout>
<@layout.registrationLayout; section>

    <#if section = "form">
        <div class="login-card" style="text-align: center;">

            <div style="margin-bottom: 1.5rem;">
                <svg width="56" height="56" viewBox="0 0 56 56" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin: 0 auto 1rem; display: block;">
                    <circle cx="28" cy="28" r="27" stroke="rgba(57,188,243,0.3)" stroke-width="2"/>
                    <circle cx="28" cy="28" r="27" stroke="rgba(57,188,243,0.15)" stroke-width="2" fill="rgba(57,188,243,0.05)"/>
                    <path d="M28 16v16" stroke="#39bcf3" stroke-width="2.5" stroke-linecap="round"/>
                    <circle cx="28" cy="38" r="2" fill="#39bcf3"/>
                </svg>
            </div>

            <h2 class="login-title" style="margin-bottom: 0.75rem;">
                <#if message?has_content>${message.summary}<#else>Something went wrong</#if>
            </h2>

            <p style="color: rgba(255,255,255,0.5); font-size: 0.9rem; margin-bottom: 2rem; line-height: 1.6;">
                <#if client?? && client.baseUrl?has_content>
                    You can go back to the application and try again.
                <#else>
                    Please close this page and try logging in again.
                </#if>
            </p>

            <#if client?? && client.baseUrl?has_content>
                <a href="${client.baseUrl}" class="login-button" style="display:block; text-decoration:none; padding: 0.95rem; text-align:center;">
                    ← Back to Application
                </a>
            </#if>

        </div>
    </#if>

</@layout.registrationLayout>
